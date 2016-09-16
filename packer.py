
import argparse
import datetime
import functools
import glob
import json

import netCDF4
import numpy as np


def pack(id, name, lr):
    # Config - Where are the source files?  What resolution to use?
    config = {
        True: {  # lr=True, use lowres tiles
            "source_path": "/short/z00/prl900/nonagg_tiles/{}/FC_LR.*.nc",
            "scale": 10},
        False: {  # lr=False, use full resolution tiles
            "source_path": "/short/z00/prl900/nonagg_tiles/{}/FC.*.nc",
            "scale": 1}
        }

    phot_stack = None
    nphot_stack = None
    bare_stack = None

    timestamps = []
    proj_wkt = None
    geot = None

    for file in sorted(glob.glob(config[lr]["source_path"].format(id))):
        tile_ts = file.split("/")[-1].split(".")[3][1:]
        date = datetime.datetime(int(tile_ts[:4]), 1, 1) + datetime.timedelta(int(tile_ts[4:])-1)
        timestamps.append(date)
   
        with netCDF4.Dataset(file, 'r', format='NETCDF4') as src:
            if geot is None:
                var = src["sinusoidal"]
                proj_wkt = var.spatial_ref
                geot = [float(val) for val in var.GeoTransform.split(" ") if val]

            if phot_stack is None:
                phot_stack = np.expand_dims(src["phot_veg"][:], axis=0)
                nphot_stack = np.expand_dims(src["nphot_veg"][:], axis=0)
                bare_stack = np.expand_dims(src["bare_soil"][:], axis=0)
            else:
                phot_stack = np.vstack((phot_stack, np.expand_dims(src["phot_veg"][:], axis=0)))
                nphot_stack = np.vstack((nphot_stack, np.expand_dims(src["nphot_veg"][:], axis=0)))
                bare_stack = np.vstack((bare_stack, np.expand_dims(src["bare_soil"][:], axis=0)))

    with netCDF4.Dataset(name, 'w', format='NETCDF4_CLASSIC') as dest:
        with open('nc_metadata.json') as data_file:
            attrs = json.load(data_file)
            for key in attrs:
                setattr(dest, key, attrs[key])
 
        setattr(dest, "date_created", datetime.datetime.utcnow().isoformat())

        t_dim = dest.createDimension("time", len(timestamps))
        x_dim = dest.createDimension("x", phot_stack.shape[2])
        y_dim = dest.createDimension("y", phot_stack.shape[1])

        var = dest.createVariable("time", "f8", ("time",))
        var.units = "seconds since 1970-01-01 00:00:00.0"
        var.calendar = "standard"
        var.long_name = "Time, unix time-stamp"
        var.standard_name = "time"
        var[:] = netCDF4.date2num(timestamps, units="seconds since 1970-01-01 00:00:00.0", calendar="standard")

        var = dest.createVariable("x", "f8", ("x",))
        var.units = "m"
        var.long_name = "x coordinate of projection"
        var.standard_name = "projection_x_coordinate"
        var[:] = np.linspace(
            geot[0],
            geot[0] + (config[lr]["scale"] * geot[1] * phot_stack.shape[2]),
            phot_stack.shape[2])

        var = dest.createVariable("y", "f8", ("y",))
        var.units = "m"
        var.long_name = "y coordinate of projection"
        var.standard_name = "projection_y_coordinate"
        var[:] = np.linspace(geot[3], geot[3]+(config[lr]["scale"]*geot[5]*phot_stack.shape[1]), phot_stack.shape[1])

        # Same variable properties for all data.
        # NB: use chunksizes = (5, 240, 240) for low-res data.
        create_data_var = functools.partial(
            dest.createVariable, datatype='i1', dimensions=("time", "y", "x"),
            fill_value=255, zlib=True, chunksizes=(1, 240, 240))

        var = create_data_var("phot_veg")
        var.long_name = "Photosynthetic Vegetation"
        var.units = '%'
        var.grid_mapping = "sinusoidal"
        var[:] = phot_stack

        var = create_data_var("nphot_veg")
        var.long_name = "Non Photosynthetic Vegetation"
        var.units = '%'
        var.grid_mapping = "sinusoidal"
        var[:] = nphot_stack

        var = create_data_var("bare_soil")
        var.long_name = "Bare Soil"
        var.units = '%'
        var.grid_mapping = "sinusoidal"
        var[:] = bare_stack

        var = dest.createVariable("sinusoidal", 'S1', ())
        var.grid_mapping_name = "sinusoidal"
        var.false_easting = 0.0
        var.false_northing = 0.0
        var.longitude_of_central_meridian = 0.0
        var.longitude_of_prime_meridian = 0.0
        var.semi_major_axis = 6371007.181
        var.inverse_flattening = 0.0
        var.spatial_ref = proj_wkt
        var.GeoTransform = "{} {} {} {} {} {}".format(geot[0], 10*geot[1], geot[2], geot[3], geot[4], 10*geot[5])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Modis Vegetation Analysis NetCDF aggregator.")
    parser.add_argument(dest="foldername", type=str, help="Input folder to pack.")
    args = parser.parse_args()

    foldername = args.foldername
    pack(foldername[2:], "FC_LRZ.v302.MCD43A4/FC_LR.v302.MCD43A4.{}.005.nc".format(foldername[2:]), True)
    #pack(foldername[2:], "FC.v302.MCD43A4/FC.v302.MCD43A4.{}.005.nc".format(foldername[2:]), False)
