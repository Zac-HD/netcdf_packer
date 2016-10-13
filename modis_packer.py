#!/usr/bin/env python3
"""A tool to pack Modis tiles into NetCDF files, aggregated by tile and year.

Planned features include reprojection (sinusoidal to WGS84) and combining
datasets (ie variables) from multiple files.

"""

import argparse
import collections
import concurrent.futures
import contextlib
import datetime
import functools
import glob
import json
import logging
import math
import os

import netCDF4
import numpy as np

# Version of the generic Modis Tile Stacker
__version__ = '0.1.0'

# Names are descriptive assuming a north-up image, and raster coords start
# at the top-left.  See http://www.gdal.org/gdal_datamodel.html
AffineGeoTransform = collections.namedtuple(
    'GeoTransform', ['origin_x', 'pixel_width', 'x_2',
                     'origin_y', 'y_4', 'pixel_height'])

RasterShape = collections.namedtuple('RasterShape', ['time', 'y', 'x'])

# Set up some simple logging, for diagnosis of large batch jobs
logging.basicConfig(
    level=logging.INFO,
    style='{',
    format='{asctime} {levelname:8} {name:12} {message}',
    datefmt='%Y-%m-%dT%H:%M:%S'
    )
log = logging.getLogger(__name__)


class PrefixAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return '{}    {}'.format(self.extra['prefix'], msg), kwargs


@contextlib.contextmanager
def log_prefix(prefix, _stack=[]):
    """Context manager that pushes prefix in before a log message."""
    _stack.append(prefix)
    global log
    __old_log = log
    log = PrefixAdapter(
        logging.getLogger(__name__), {'prefix': ':'.join(_stack)})
    try:
        yield
    finally:
        _stack.pop()
        log = __old_log


def log_prefix_decorator(func):
    @functools.wraps(func)
    @log_prefix('func=' + func.__name__)
    def wrapper(*args, **kwds):
        return func(*args, **kwds)  
    return wrapper


def fname_metadata(path, date_fmt=None):
    """Extract file properties from the path (ie fname).
    This function tries to isolate non-portable assumptions about filenames
    for easier porting to other datasets.
    """
    delim, parts = '.', ('dataset', 'date', 'tile', 'resolution', 'name')
    fname_parts = os.path.basename(path).split(delim, maxsplit=len(parts)-1)
    metadata = collections.namedtuple('fname_meta', parts)
    split = dict(zip(parts, fname_parts))
    split['date'] = datetime.datetime.strptime(split['date'], date_fmt)
    return metadata(**split)


@log_prefix_decorator
def get_tile_data(fname):
    """Return file data and metadata as dictionaries."""
    assert fname.endswith('.nc')
    data, sinusoidal, metadata = {}, {}, {}
    with netCDF4.Dataset(fname, format='NETCDF4') as src:
        log.debug('Opened {} for reading'.format(fname))
        assert 'time' not in src.dimensions
        for name, attr in src.variables.items():
            # Skip all but the data-carrying variables
            if name in src.dimensions:
                continue
            if name == 'sinusoidal':
                for k in ('grid_mapping_name', 'false_easting',
                          'false_northing', 'longitude_of_central_meridian',
                          'longitude_of_prime_meridian', 'semi_major_axis',
                          'inverse_flattening', 'spatial_ref', 'GeoTransform'):
                    sinusoidal[k] = getattr(attr, k)
                continue
            metadata[name] = {k: getattr(attr, k) for k in
                              ('long_name', 'units', 'grid_mapping', 'dtype')}
            data[name] = np.copy(attr)
            data[name][attr == getattr(attr, '_FillValue', np.nan)] = np.nan
    return data, metadata, sinusoidal


@log_prefix_decorator
def project_array_to_latlon(array, geot, wkt_str, out_res_degrees=0.005):
    """Reproject a tile from Modis Sinusoidal to WGS84 Lat/Lon coordinates.
    Metadata is handled by the calling function.
    """
    from osgeo import gdal, gdal_array, osr
    assert isinstance(geot, AffineGeoTransform)

    def array_to_raster(array, geot, wkt):
        ysize, xsize = array.shape  # unintuitive order, but correct!
        dataset = gdal.GetDriverByName('MEM').Create(
            '', xsize, ysize,
            eType=gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype))
        dataset.SetGeoTransform(geot)
        dataset.SetProjection(wkt)
        dataset.GetRasterBand(1).WriteArray(array)
        return dataset

    input_data = array_to_raster(array, geot, wkt_str)

    # Set up the reference systems and transformation
    from_sr = osr.SpatialReference()
    from_sr.ImportFromWkt(wkt_str)
    to_sr = osr.SpatialReference()
    to_sr.SetWellKnownGeogCS("WGS84")
    tx = osr.CoordinateTransformation(from_sr, to_sr)

    # Get all corners in new proj, and determine the bounding box
    lrx = geot.origin_x + geot.pixel_width * input_data.RasterXSize
    lry = geot.origin_y + geot.pixel_height * input_data.RasterYSize
    xs, ys, _ = zip(*(tx.TransformPoint(*coord) for coord in [
        (geot.origin_x, geot.origin_y), (geot.origin_x, lry),
        (lrx, geot.origin_y), (lrx, lry)]))

    new_geot = AffineGeoTransform(
        min(xs), out_res_degrees, 0, max(ys), 0, -out_res_degrees)
    def arr_size(ds):
        return math.ceil((max(ds) - min(ds)) / out_res_degrees)
    dest_arr = np.empty((arr_size(ys), arr_size(xs)))
    dest_arr[:] = np.nan
    dest = array_to_raster(dest_arr, new_geot, to_sr.ExportToWkt())

    # Perform the projection/resampling
    gdal.ReprojectImage(
        input_data, dest,
        from_sr.ExportToWkt(), to_sr.ExportToWkt(),
        gdal.GRA_Bilinear)
    return dest.GetRasterBand(1).ReadAsArray(), new_geot


@log_prefix_decorator
def write_stacked_netcdf(*, filename, timestamps, data, metadata, sinusoidal,
                         attributes, chunk_shape, geot):
    """Write a big file."""
    # Calculate some helpers
    raster_size = RasterShape(*data[next(iter(metadata))].shape)
    log.debug(raster_size)
    # Write the file
    with netCDF4.Dataset(filename, 'w', format='NETCDF4_CLASSIC') as dest:
        # Set top-level file attributes and metadata
        for key, val in attributes.items():
            setattr(dest, key, val)
        setattr(dest, "date_created", datetime.datetime.utcnow().isoformat())

        # Create dimensions and corresponding variables
        dest.createDimension("time", len(timestamps))
        dest.createVariable("time", "f8", ("time",))
        dest['time'].units = "seconds since 1970-01-01 00:00:00.0"
        dest['time'].calendar = "standard"
        dest['time'].long_name = "Time, unix time-stamp"
        dest['time'].standard_name = "time"
        dest['time'][:] = timestamps

        dest.createDimension("x", raster_size.x)
        dest.createVariable("x", "f8", ("x",))
        dest['x'].units = "m"
        dest['x'].long_name = "x coordinate of projection"
        dest['x'].standard_name = "projection_x_coordinate"
        dest['x'][:] = np.linspace(
            start=geot.origin_x,
            stop=geot.origin_x + geot.pixel_width * raster_size.x,
            num=raster_size.x)

        dest.createDimension("y", raster_size.y)
        dest.createVariable("y", "f8", ("y",))
        dest['y'].units = "m"
        dest['y'].long_name = "y coordinate of projection"
        dest['y'].standard_name = "projection_y_coordinate"
        dest['y'][:] = np.linspace(
            start=geot.origin_y,
            stop=geot.origin_y + geot.pixel_height * raster_size.y,
            num=raster_size.y)

        if sinusoidal is not None:
            # The sinusoidal variable is somewhat special; attrs saved above
            # But not used if data is reprojected to WGS84
            dest.createVariable("sinusoidal", 'S1', ())
            for name, val in sinusoidal.items():
                setattr(dest['sinusoidal'], name, val)
        else:
            # TODO:  set up correct grid mapping for WGS84
            pass

        def make_var(dest, name, attrs, cube):
            """Create a data variable, add data, and set attributes."""
            var = dest.createVariable(
                name, attrs.pop('dtype'), dimensions=("time", "y", "x"),
                zlib=bool(chunk_shape), chunksizes=chunk_shape)
            for key, value in attrs.items():
                setattr(var, key, value)
            var[:] = cube

        for name, attrs in metadata.items():
            make_var(dest, name, attrs, data[name])


def stack_tiles(ts_fname_list, *,
                out_file, attributes, chunk_shape, reproject):
    """Stack the list of input tiles, according to the given arguments."""
    assert ts_fname_list, 'List of tiles to stack has contents'
    data = {}
    for timestamp, file in sorted(ts_fname_list):
        # When combining files (eg LFMC + Flammability), unpack and assign
        # retvals separately... and be careful to validate consistency etc.
        data[timestamp], metadata, sinusoidal = get_tile_data(file)
        assert metadata, 'Input files must have data variables'
        assert sinusoidal, 'MODIS tiles must have a sinusoidal variable'

    geot = AffineGeoTransform(
        *[float(v) for v in sinusoidal['GeoTransform'].split()])
    if reproject:
        for name in metadata:
            for timestamp in data:
                data[timestamp][name], new_geot = project_array_to_latlon(
                    data[timestamp][name], geot, sinusoidal['spatial_ref'])
            # TODO: is this correct??  See writer function note and replace it
            metadata[name].pop('grid_mapping', None)
        geot = new_geot
        sinusoidal = None

    # Write out the aggregated thing
    write_stacked_netcdf(
        filename=out_file, timestamps=[ts for ts, _ in ts_fname_list],
        data={name: np.stack([dat[name] for dat in data.values()], axis=0)
              for name in metadata},
        metadata=metadata, sinusoidal=sinusoidal,
        attributes=attributes, chunk_shape=chunk_shape, geot=geot)


def checkpointer(ts_fname_list, args):
    """Skip work that has already been done, precalc filenames, etc.
    This function is submitted to the process pool, so we also set logging 
    level and prefix to avoid defaults and shared stacks (!).
    """
    out_file = os.path.basename(ts_fname_list[0][1])
    meta = fname_metadata(out_file, args.strptime_fmt)
    out_file = os.path.join(args.output_dir, out_file.replace(
        meta.date.strftime(args.strptime_fmt), str(meta.date.year)))
    if args.as_wgs84:
        out_file = os.path.join(os.path.dirname(out_file),
                                'WGS84.' + os.path.basename(out_file))
    # Reset log prefix stack, and start fresh with tile and year
    log.setLevel(args.loglevel)  # also set this for each process...
    with log_prefix('tile={}:year={}'.format(meta.tile, meta.date.year)):
        log.info('Output path is ' + out_file)
        if os.path.isfile(out_file):
            if os.path.getsize(out_file) == 0:
                # Writing was previously attempted, but wasn't finished
                log.debug('Removing empty file, will retry stacking')
                os.remove(out_file)
            elif not args.ignore_checkpoints:
                log.info('File already exists, nothing more to do here.')
                return
        # Call the stacker, with just the relevant bits of the args namespace
        stack_tiles(ts_fname_list, out_file=out_file,
                    attributes=args.metadata, chunk_shape=args.compress_chunk)


def main():
    """Run the embarrassingly parrellel worker function (stack_tiles)."""
    # Get shell arguments, which are pre-validated and transformed (below)
    args = get_validated_args()
    log.setLevel(args.loglevel)
    log.debug('Got the args')

    # Group input files by tile ID string and year
    grouped_files = collections.defaultdict(list)
    for afile in args.files:
        meta = fname_metadata(afile, args.strptime_fmt)
        grouped_files[(meta.tile, meta.date.year)].append(
            (meta.date.timestamp(), afile))

    # Stack tiles, using multiprocess only if jobs > 1 for nicer tracebacks
    func = functools.partial(checkpointer, args=args)
    pool_size = min([len(grouped_files), args.jobs])
    if pool_size > 1:
        log.info('Using pool of %i processes; errors suppressed', pool_size)
        with concurrent.futures.ProcessPoolExecutor(args.jobs) as executor:
            executor.map(func, grouped_files.values())
    else:
        log.info('Using serial execution mode')
        list(map(func, grouped_files.values()))


def get_validated_args():
    """Handle command-line arguments, including default values.

    The goal is to make argparse do as much of the work as possible, eg with
    type validation functions, to 'fail fast' and simplify other code.
    This gives much better user feedback in the calling shell, too.
    """
    def glob_arg_type(val):
        """Validate arg and transform glob pattern to file list."""
        files = tuple(os.path.normpath(p) for p in glob.glob(val))
        if not files:
            raise argparse.ArgumentTypeError('glob pattern matchs no files')
        return files

    def dir_arg_type(val):
        """Validate that arg is an existing directory."""
        if not os.path.exists(val):
            raise argparse.ArgumentTypeError('path does not exist')
        if not os.path.isdir(val):
            raise argparse.ArgumentTypeError('path is not a directory')
        return val

    def nc_meta_type(fname):
        """Load attributes from json file with useful errors."""
        if not os.path.isfile(fname):
            raise argparse.ArgumentTypeError(fname + ' does not exist')
        try:
            with open(fname) as file_handle:
                attrs = json.load(file_handle)
        except:
            raise argparse.ArgumentTypeError(
                'could not load attributes from ' + fname)
        # TODO:  check attributes match relevant conventions
        return attrs

    def chunksize_type(string):
        """Validate arg and transform string to 3-tuple representing size."""
        if string == '0':
            return ()
        try:
            size = [int(n) for n in string.strip('()').split(',')]
        except Exception:
            raise argparse.ArgumentTypeError(
                'could not parse chunksize to integers')
        try:
            size = RasterShape(*size)
        except TypeError:
            raise argparse.ArgumentTypeError(
                'chunksize must be three dimensional (time, Y, X)')
        if not all(n >= 1 for n in size):
            raise argparse.ArgumentTypeError(
                'all dimensions of chunksize must be natural numbers')
        return size

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-V', '--version', action='version', version=__version__)

    parser.add_argument(
        'files', type=glob_arg_type,
        help='glob pattern for the file(s) to process')
    parser.add_argument(
        '--output-dir', type=dir_arg_type, default='.', metavar='DIR',
        help='directory for output files. default="%(default)s"')

    parser.add_argument(
        '--metadata', type=nc_meta_type, metavar='FILE',
        default=nc_meta_type('./nc_metadata.json'),
        help='path to netCDF attributes file (default ./nc_metadata.json)')
    parser.add_argument(
        '--strptime_fmt', default='A%Y%j', metavar='FMT',
        help='Date format in filenames. default=%(default)s')

    parser.add_argument(
        '--compress-chunk', default=chunksize_type('1,240,240'),
        metavar='T,Y,X', type=chunksize_type, help=(
            'Chunk size for compression as comma-seperated integers, '
            'default=%(default)s.  The default is tuned for 2D display; '
            'increase time-length of chunk if time-dimension access is '
            'common.  "0" disables compression.'))
    parser.add_argument(
        '--ignore-checkpoints', action='store_true',
        help='Do not skip existing output files - more expensive than '
        'default behaviour, but allows replacment of "expired" data.  '
        'Partial files are *not* detected and would otherwise be skipped.')
    parser.add_argument(
        '--as-wgs84', action='store_true',
        help='Reproject data to WGS84 lat/lon as well as aggregating.')

    parser.add_argument(
        '--jobs', default=os.cpu_count() or 1, type=int,
        help='How many processes to use.  default: num_CPUs == %(default)s')

    log_grp = parser.add_mutually_exclusive_group()
    log_grp.add_argument(
        '--quiet', action="store_const", dest="loglevel", 
        help='Only log warnings or errors, not informational messages',
        const=logging.WARNING, default=logging.INFO)
    log_grp.add_argument(
        '--verbose', action="store_const", dest="loglevel", 
        help='Log all messages, including debug information',
        const=logging.DEBUG)

    # Parse args, and check that osgeo (GDAL) is available if reprojecting
    args = parser.parse_args()
    if args.as_wgs84:
        try:
            import osgeo
            assert osgeo
        except ImportError:
            parser.error('Cannot reproject data without GDAL (osgeo)')
    return args


if __name__ == '__main__':
    main()
