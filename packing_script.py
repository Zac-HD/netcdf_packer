import collections
import glob
import os
import datetime

def getMeta(path):
    filename = os.path.basename(path)
    dataset, date_str, tile, resolution, name = filename.split('.', maxsplit=4)
    return {'date': datetime.datetime.strptime(date_str, 'A%Y%j'), 'tile': tile}

def getModisFileDict(files):
    org_files = collections.defaultdict(list)
    for afile in files:
        meta = getMeta(afile)
        org_files[(meta['tile'], meta['date'].year)] += [afile]
    return org_files


if __name__ == "__main__":
    files = glob.glob("/g/data1/xc0/project/FMC_Australia/FMC_PRODUCTS_AU/FMC_Product_using_MCD12Q1/*.nc")
    print(getModisFileDict(files))
    
