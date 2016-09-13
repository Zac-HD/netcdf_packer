import glob
import os
import datetime

def getMeta(path):
    filename = os.path.basename(path).replace('.nc', '')
    dataset, date_str, tile, resolution, create_date = filename.split('.')
    return {'date': datetime.datetime.strptime(date_str, 'A%Y%j'), 'tile': tile}

def getModisFileDict(files):
    org_files = {}
    for afile in files:
        meta = getMeta(afile)
        tile, year = meta['tile'], meta['date'].year
        if tile not in org_files:
            org_files[tile] = {}
        org_files[tile][year] = org_files[tile].get(year, []) + [afile]
        
    return org_files


if __name__ == "__main__":
    files = glob.glob("/g/data1/xc0/project/FMC_Australia/FMC_PRODUCTS_AU/FMC_Product_using_MCD12Q1/*.nc")
    print(getModisFileDict(files))
    
