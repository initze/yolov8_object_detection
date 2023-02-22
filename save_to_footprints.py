import geopandas as gpd
from pathlib import Path
import pandas as pd
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
args = parser.parse_args()


def main():
    # setup paths
    ds_name = args.name
    save_dir = Path('output') / ds_name

    # load inference results
    df_class_count = pd.read_csv(save_dir / 'detected_image_summary.csv')
    
    # load vectors
    vector_dir = Path('/isipd/projects/Response/Restricted_Airborne/MACS/2021_Perma-X_Alaska/01_raw_data')
    vector_file = vector_dir / ds_name / f'{ds_name}_footprints_full.shp'
    gdf = gpd.read_file(vector_file)

    # extract basename
    gdf['image_id'] = gdf['Basename'].str.replace('.macs', '')
    # join (left)
    joined = gdf.set_index('image_id').join(df_class_count.set_index('image_id'))

    # setup output columns
    cols = list(df_class_count.columns.values)
    cols.append('geometry')
    
    gdf_out = joined.reset_index(drop=False)[cols].replace(np.nan, 0)

    # save files
    outfile = save_dir / (ds_name+ '_vector.gpkg')
    gdf_out.to_file(outfile)

    # calculate centroids and save
    gdf_out_centroid = gdf_out.copy()
    gdf_out_centroid['geometry'] = gdf_out.centroid

    outfile_centroid = save_dir / (ds_name+ '_vector_centroid.gpkg')
    gdf_out_centroid.to_file(outfile_centroid)

if __name__=='__main__':
    main()