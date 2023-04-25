from pathlib import Path

from leafmap import leafmap
from tqdm import tqdm

from utils import *

import os

import pandas as pd
import geopandas as gpd
import numpy as np

def image_tiling(infile, target_dir_base, tilesize=512, overlap_factor=2**3, dry_run=False):
    # tiling preparation
    xsize = ysize = tilesize
    overlap = int(xsize / overlap_factor)
    outname = infile.stem + '_'+ str(xsize)
    outdir = target_dir_base / outname
    # tiling run
    s = f'gdal_retile.py -of GTiff -overlap {overlap} -ot Byte -ps {xsize} {ysize} -targetDir {outdir} {infile}'
    print(s)
    # check if exists
    outdir_exists = outdir.exists()
    if not outdir_exists:
        if not dry_run:
            os.makedirs(outdir, exist_ok=True)
            os.system(s)
    else:
        print(f'{outdir} already exists! \nSkipped tiling!\n')
    return outdir


def read_labels(labels_dir, outdir):
    flist_labels = list(labels_dir.glob('*.txt'))
    df_list = []
    for fl in flist_labels[:]:
        stem = fl.stem
        raster_file = Path(outdir) / (stem+'.tif')
        columns = ['class', 'x', 'y', 'x2', 'y2', 'confidence'] 
        df = pd.read_table(fl, header=None, sep=' ')
        df.columns=columns
        df['raster_file'] = raster_file
        df['geometry'] = df.apply(row_to_geom, axis=1)
        df_list.append(df)

    df_merge = pd.concat(df_list).reset_index()
    return df_merge


def run_yolo_detection(MODEL_BASE_DIR, TILESIZE, PREDICTION_MODEL, outdir, model_name=None, dry_run=False):
    if not model_name: 
        outname = outdir.stem
    else:
        outname = f'{model_name}/{outdir.stem}'

    print(outname)
    prediction_dir = MODEL_BASE_DIR / 'runs' / 'detect' / outname
    prediction_dir_exists = prediction_dir.exists()
    if not prediction_dir_exists:
        s_yolo = f"yolo predict model={PREDICTION_MODEL} source={outdir} conf=0.1 save=False imgsz={TILESIZE} device=\'7\' save_txt=True save_conf=True max_det=500 name={outname}"
        #print(s_yolo)
        if not dry_run:
            os.system(s_yolo)
    else:
        print(f'{prediction_dir} already exists! \nSkipped processing!\n')
    return outname


def run_results_to_vector(MODEL_BASE_DIR, outdir, outname):
    prediction_dir = MODEL_BASE_DIR / 'runs' / 'detect' / outname
    vector_file = prediction_dir / (Path(outname).stem +'.gpkg')
    vector_file_exists = vector_file.exists()
    if not vector_file_exists:
        #print(prediction_dir)
        labels_dir = prediction_dir / 'labels'
            #print(labels_dir)
        df_merge = read_labels(labels_dir, outdir)
        gdf = gpd.GeoDataFrame(df_merge, crs='EPSG:3857', geometry=df_merge.geometry)
        gdf.drop(columns=['raster_file']).to_file(vector_file)
    else:
        print(f'{vector_file} already exists! \nSkipped Vector creation!\n')

def main():
    # KEY SETTINGS
    MODEL_BASE_DIR = Path('.')
    TILESIZE = 512
    INPUTFILE_DIR = MODEL_BASE_DIR / 'data_download'
    TILE_DIR = MODEL_BASE_DIR / 'data_satellite'
    regex = 'ESRI_Satellite'#'ESRISatellite'
    flist = list(INPUTFILE_DIR.glob(f'*{regex}*.tif'))
    print(flist)
    PREDICTION_MODEL = Path('../MACS_UHR_DL/Beavers/v13i.yolov5pytorch_yolov8l_lr005_mom090-train_continue-v123/weights/best.pt')
    MODEL_NAME = 'v13i.yolov5pytorch_yolov8l_lr005_mom090-train_continue-v123'
    
    # loop over all datasets
    for infile in tqdm(flist):
        # image_tiling
        outdir = image_tiling(infile, target_dir_base=TILE_DIR, tilesize=TILESIZE, dry_run=False)
        #"""
        # YOLOv8 run
        outname = run_yolo_detection(MODEL_BASE_DIR, TILESIZE, PREDICTION_MODEL, outdir, model_name=MODEL_NAME, dry_run=False)
        # vector output creation
        run_results_to_vector(MODEL_BASE_DIR, outdir, outname)
        #"""

if __name__=="__main__":
    main()
