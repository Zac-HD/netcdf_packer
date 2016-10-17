# netcdf_packer
Python scripts for packaging netcdf4

Call `python modis_packer.py --help` for usage information,
or see the module docstring.

The required packages can be installed with
`conda create -n modis-packer numpy netcdf4 gdal`, with `gdal` only
required for the optional reprojection to WGS84 lat/lon.
