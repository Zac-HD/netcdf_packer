#!/usr/bin/env python3
"""A tool to pack Modis tiles into NetCDF files, aggregated by tile and year.
"""

import argparse
import collections
import concurrent.futures
import datetime
import functools
import glob
import json
import os

import netCDF4
import numpy as np

# Version of the generic Modis Tile Stacker
__version__ = '0.0.1'

# Names are descriptive assuming a north-up image, and raster coords start
# at the top-left.  See http://www.gdal.org/gdal_datamodel.html
AffineGeoTransform = collections.namedtuple(
    'GeoTransform', ['origin_x', 'pixel_width', 'x_2',
                     'origin_y', 'pixel_height', 'y_5'])
# An XY coordinate in pixels
RasterShape = collections.namedtuple('Shape', ['time', 'y', 'x'])


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


def get_tile_data(fname, metadata, data_ts, sinusoidal):
    """Add file data to the metadata and data dictionaries."""
    # Yep, a function which only returns by mutating it's arguments...
    assert fname.endswith('.nc')
    with netCDF4.Dataset(fname, format='NETCDF4') as src:
        assert 'time' not in src.dimensions
        # Save metadata every time (low-cost, easier reading)
        for name, attr in src.variables.items():
            # Skip all but the data-carrying variables
            if name in src.dimensions:
            #('x', 'y', ):
                continue
            if name == 'sinusoidal':
                for k in ('grid_mapping_name', 'false_easting',
                          'false_northing', 'longitude_of_central_meridian',
                          'longitude_of_prime_meridian', 'semi_major_axis',
                          'inverse_flattening', 'spatial_ref', 'GeoTransform'):
                    sinusoidal[k] = getattr(attr, k)
                continue
            # Save metadata every time (low-cost, easier reading)
            for k in ('long_name', 'units', 'grid_mapping', 'dtype'):
                metadata[name][k] = getattr(attr, k)
            # No ordered defaultdict class, and order is more important
            data_ts[name] = np.copy(attr)


def write_stacked_netcdf(
        *, filename, args, timestamps, data, metadata, sinusoidal):
    """Write a big file."""
    # Calculate some helpers
    raster_size = RasterShape(*data[next(iter(metadata))].shape)
    geot = AffineGeoTransform(
        *[float(v) for v in sinusoidal['GeoTransform'].split()])
    # Write the file
    with netCDF4.Dataset(filename, 'w', format='NETCDF4_CLASSIC') as dest:
        # Set top-level file attributes and metadata
        for key, val in args.metadata.items():
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

        # The sinusoidal variable is somewhat special; attrs saved above
        dest.createVariable("sinusoidal", 'S1', ())
        for name, val in sinusoidal.items():
            setattr(dest['sinusoidal'], name, val)

        def make_var(dest, name, attrs, cube):
            """Create a data variable, add data, and set attributes."""
            var = dest.createVariable(
                name, attrs.pop('dtype'), dimensions=("time", "y", "x"),
                zlib=bool(args.compress_chunk), chunksizes=args.compress_chunk)
            for key, value in attrs.items():
                setattr(var, key, value)
            var[:] = cube

        for name, attrs in metadata.items():
            make_var(dest, name, attrs, data[name])


def stack_tiles(fname_list, *, args):
    """Stack the list of input tiles, according to the given arguments."""
    sinusoidal = {}
    metadata = collections.defaultdict(dict)
    data = collections.OrderedDict()
    for timestamp, file in sorted(
            (fname_metadata(f, args.strptime_fmt).date.timestamp(), f)
            for f in fname_list):
        data[timestamp] = data.get(timestamp) or {}
        get_tile_data(file, metadata, data[timestamp], sinusoidal)

    # check that we did actually read something
    assert metadata, 'Input files must have data variables'
    assert sinusoidal, 'MODIS tiles must have a sinusoidal variable'

    out_file = os.path.join(
        args.output_dir, '{0}.{1.year}.{2}.{3}.{4}'.format(
            *fname_metadata(os.path.basename(fname_list[0]),
                            args.strptime_fmt)))
    # Write out the aggregated thing
    write_stacked_netcdf(
        filename=out_file, args=args, timestamps=tuple(data),
        data={name: np.stack([dat[name] for dat in data.values()], axis=0)
              for name in metadata},
        metadata=metadata, sinusoidal=sinusoidal)


def main():
    """Run the embarrassingly parrellel worker function (stack_tiles)."""
    # Get shell arguments, which are pre-validated and transformed (below)
    args = get_validated_args()

    # Group input files by tile ID string and year
    grouped_files = collections.defaultdict(list)
    for afile in args.files:
        meta = fname_metadata(afile, args.strptime_fmt)
        grouped_files[(meta.tile, meta.date.year)].append(afile)

    # Stack tiles, using multiprocess only if jobs > 1 for nicer tracebacks
    func = functools.partial(stack_tiles, args=args)
    if args.jobs > 1:
        with concurrent.futures.ProcessPoolExecutor(args.jobs) as executor:
            executor.map(func, grouped_files.values())
    else:
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
        '--output_dir', type=dir_arg_type, default='.', metavar='DIR',
        help='directory for output files. default="%(default)s"')

    parser.add_argument(
        '--metadata', type=nc_meta_type, metavar='FILE',
        default=nc_meta_type('./nc_metadata.json'),
        help='path to netCDF attributes file (default ./nc_metadata.json)')
    parser.add_argument(
        '--strptime_fmt', default='A%Y%j', metavar='FMT',
        help='Date format in filenames. default=%(default)s')

    parser.add_argument(
        '--compress_chunk', default=chunksize_type('1,240,240'),
        metavar='T,Y,X', type=chunksize_type, help=(
            'Chunk size for compression as comma-seperated integers, '
            'default=%(default)s.  The default is tuned for 2D display; '
            'increase time-length of chunk if time-dimension access is '
            'common.  "0" disables compression.'))

    parser.add_argument(
        '--jobs', default=os.cpu_count() or 1, type=int,
        help='How many processes to use.  default: num_CPUs == %(default)s')

    return parser.parse_args()


if __name__ == '__main__':
    main()
