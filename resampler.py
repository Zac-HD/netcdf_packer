import argparse
import os.path
import netCDF4
import numpy as np
import scipy.ndimage

def netcdf_downsampling(source, zoom, dest):
    with netCDF4.Dataset(source, 'r', format='NETCDF4') as src:
        with netCDF4.Dataset(dest, 'w', format='NETCDF4') as dst:
            
            # Copy global attributes
            for item, value in src.__dict__.items():
                dst.__setattr__(item, value)
            
            # Copy dimensions
            for dim in src.dimensions:
                if dim == 'x' or dim == 'y':
                    dst.createDimension(dim, len(src.dimensions[dim])*zoom)
                else:
                    dst.createDimension(dim, len(src.dimensions[dim]))
                    
            # Copy variables with attributes
            for name in src.variables:
                
                # Create variables
                new_var = None
                if hasattr(src.variables[name], "_FillValue"):
                    new_var = dst.createVariable(name, src.variables[name].datatype, src.variables[name].dimensions, fill_value=src.variables[name]._FillValue)
                else:
                    new_var = dst.createVariable(name, src.variables[name].datatype, src.variables[name].dimensions)
                    
                # Copy content of variables (resampling if needed)
                if src.variables[name].dimensions == (u'time', u'y', u'x',):
                    new_var[:] = scipy.ndimage.zoom(src.variables[name][:], [1, zoom, zoom], mode='nearest')
                elif src.variables[name].dimensions == (u'x',) or src.variables[name].dimensions == (u'y',):
                    new_var[:] = src.variables[name][::int(1/zoom)]
                elif src.variables[name].dimensions == (u'time',):
                    new_var[:] = src.variables[name][:]
                else:
                    new_var[:] = src.variables[name][:].data
                  
                # Copy attributes
                for item, value in src.variables[name].__dict__.items():
                    # Modify Geotransform attribute with new pixel size
                    if name == "sinusoidal" and item == "GeoTransform":
                        geot_values = value.split()
                        geot_values[1] = str(float(geot_values[1])/zoom)
                        geot_values[5] = str(float(geot_values[5])/zoom)
                        new_var.__setattr__(item, " ".join(geot_values) + " ")
                        continue

                    if not item.startswith('_'):
                        new_var.__setattr__(item, value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""NetCDF4 file resample parameters""")
    parser.add_argument(dest="src", type=str, help="Full path to source.")
    parser.add_argument(dest="zoom", type=float, help="Zoom level to apply to x and y dimensions.")
    parser.add_argument(dest="dst", type=str, help="Full path to destination.")
    args = parser.parse_args()

    if not os.path.isfile(args.dst):
        netcdf_downsampling(args.src, args.zoom, args.dst)
