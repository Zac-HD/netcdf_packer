import glob
import numpy as np
from osgeo import gdal

ds = gdal.Open("/g/data/xc0/project/FMC_Australia/FMC_PRODUCTS_AU/Flammability index product/FI_2005121_h31v12.tif")
print(ds.GetGeoTransform())

arr = ds.ReadAsArray()

print(arr.shape, arr.dtype)

""" Function for downsampling
resampled = scipy.ndimage.zoom(arr, [.1, .1, 1])
"""


"""
files = [a + '/' + a.split('/')[-1] + '_B1.TIF' for a in glob.glob('./LS8_095_073/*')]

lons_start = []
lons_end = []
lats_start = []
lats_end = []

for f in files:
    ds = gdal.Open(f)
    lons_start.append(ds.GetGeoTransform()[0])
    lons_end.append(ds.GetGeoTransform()[0] + ds.GetGeoTransform()[1]*ds.RasterXSize)
    
    lats_start.append(ds.GetGeoTransform()[3])
    lats_end.append(ds.GetGeoTransform()[3] + ds.GetGeoTransform()[5]*ds.RasterYSize)
    
print max(lats_start), min(lats_end)
print min(lons_start), max(lons_end)

lons = np.arange(287985.0, 520515.0 + 30., 30.0)
lats = np.arange(-1961685.0, -2194215.0 - 30., -30.0)

cube = None

for f in files:
    canvas = np.zeros((len(lats), len(lons)), dtype=np.int16)

    ds = gdal.Open(f)
    geot = ds.GetGeoTransform()
    lat_idx = (np.abs(lats-geot[3])).argmin()
    lon_idx = (np.abs(lons-geot[0])).argmin()
    
    #print "[{}:{}, {}:{}]".format(lat_idx, lat_idx+ds.RasterYSize, lon_idx, lon_idx+ds.RasterXSize)
    #print ds.ReadAsArray().shape
    
    canvas[lat_idx:lat_idx+ds.RasterYSize, lon_idx:lon_idx+ds.RasterXSize] = ds.ReadAsArray()

    if cube is None:
        cube = canvas
    else:
        cube = np.dstack((cube, canvas))
    
        
print cube.shape
"""
