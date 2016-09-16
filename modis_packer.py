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


# Version of the generic Modis Tile Stacker
__version__ = '0.0.1'


def metadata_from_input_filename(path, date_fmt):
    """Extract file properties from the path (ie fname).
    This function tries to isolate non-portable assumptions about filenames
    for easier porting to other datasets.
    """
    # Fname part delimiter, and list of tokens in order
    delim = '.'
    parts = ('dataset', 'date', 'tile', 'resolution', 'name')
    # Construct and return a namedtuple of metadata
    fname_parts = os.path.basename(path).split(delim, maxsplit=len(parts)-1)
    metadata = collections.namedtuple('fname_meta', parts)
    split = dict(zip(parts, fname_parts))
    split['date'] = datetime.datetime.strptime(date_fmt, split['date'])
    return metadata(**split)


def stack_tiles(fname_list, args):
    """Stack the list of input tiles, according to the given arguments."""
    # See packer.py; reimplement with scipy.io.netcdf if possible so we
    # can use the standard Anaconda stack
    raise NotImplementedError


def main():
    """Run the embarrassingly parrellel worker function (stack_tiles)."""
    # Get shell arguments, which are pre-validated and transformed (below)
    args = get_validated_args()

    # Group input files by tile ID string and year
    grouped_files = collections.defaultdict(list)
    for afile in args.files:
        meta = metadata_from_input_filename(afile, args.strptime_fmt)
        grouped_files[(meta.tile, meta.date.year)] += [afile]

    # Stack tiles, using multiprocess only if jobs > 1 for nicer tracebacks
    func = functools.partial(stack_tiles, args)
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
    ATE = argparse.ArgumentTypeError

    def glob_arg_type(val):
        files = tuple(os.path.normpath(p) for p in glob.glob(val))
        if not files:
            raise ATE('glob pattern matchs no files')
        return files

    def dir_arg_type(val):
        if not os.path.exists(val):
            raise ATE('path does not exist')
        if not os.path.isdir(val):
            raise ATE('path is not a directory')
        return val

    def nc_meta_type(fname):
        if not os.path.isfile(fname):
            raise ATE(fname + ' does not exist')
        try:
            with open(fname) as file_handle:
                attrs = json.load(file_handle)
        except:
            raise ATE('could not load attributes from ' + fname)
        # TODO:  verify or sanity-check attributes here
        return attrs

    def chunksize_type(string):
        if string == '0':
            return ()
        try:
            size = tuple(int(n) for n in string.strip('()').split(','))
        except Exception:
            raise ATE('could not parse chunksize to integers')
        if len(size) != 3:
            raise ATE('chunksize must be three dimensional (time, X, Y)')
        if not all(n <= 1 for n in size):
            raise ATE('all dimensions of chunksize must be positive integers')
        return size

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-V', '--version', action='version', version=__version__)

    parser.add_argument(
        'input_glob', type=glob_arg_type, dest='files',
        help='glob pattern for the file(s) to process')
    parser.add_argument(
        '--output_dir', type=dir_arg_type, default='.', metavar='DIR',
        help='directory for output files. default=%(default)s')

    parser.add_argument(
        '--metadata', type=nc_meta_type, metavar='FILE',
        default=nc_meta_type('./nc_metadata.json'),
        help='path to netCDF attributes file (default ./nc_metadata.json)')
    parser.add_argument(
        '--strptime_fmt', default='A%Y%j', metavar='FMT',
        help='Date format in filenames. default=%(default)s')

    parser.add_argument(
        '--fill_value', type=int, metavar='INT',
        help='Fill value of data, if an integer.  NaN (float) if not given.')
    parser.add_argument(
        '--compress_chunk', default=(1, 240, 240), metavar='T,X,Y',
        type=chunksize_type, help=(
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
