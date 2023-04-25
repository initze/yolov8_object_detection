from pathlib import Path
from leafmap import leafmap
from tqdm import tqdm
import geopandas as gpd
import numpy as np
from utils import *

def main():

    LON_START = -163.0
    LON_END = -162.0
    LAT_START = 67.6
    LAT_END = 67.7
    GRID_SIZE = 0.1

    # KEY SETTINGS
    DOWNLOAD = True
    SERVICE = 'ESRI_Satellite'
    zoomlevel = 18
    quiet_download = True
    to_cog = True

    # Auto create grid
    lons = np.arange(LON_START, LON_END, GRID_SIZE)
    lats = np.arange(LAT_START, LAT_END, GRID_SIZE)
    lon_grid, lat_grid = np.around(np.meshgrid(lons, lats), decimals=2)
    coordinates = np.array([lon_grid.ravel(), lat_grid.ravel()]).T

    if SERVICE == 'ESRI_Satellite':
        source = 'http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}' # ESRI Satellite
    elif SERVICE == 'Google_Satellite':
        source = 'SATELLITE'

    for lon, lat in tqdm(coordinates[:]):
        box = list(np.around([lon, lat, lon+GRID_SIZE, lat+GRID_SIZE], decimals=2))
        lon_min, lat_min, lon_max, lat_max = box
        site_name = f'{lon_min}_{lon_max}_{lat_min}_{lat_max}'
        kwargs_tms_to_geotiff = dict(quiet=quiet_download, to_cog=to_cog, bbox=box)
        out_file = Path('data_download') / f'{SERVICE}_{site_name}_{zoomlevel}.tif'
        if DOWNLOAD:
            if not out_file.exists():
                leafmap.tms_to_geotiff(out_file.as_posix(), zoom=zoomlevel, source=source, **kwargs_tms_to_geotiff)
            else:
                print(f'Outfile {out_file} already exists!')

if __name__=="__main__":
    main()
