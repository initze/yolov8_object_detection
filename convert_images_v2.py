import os
import tqdm
import numpy as np
from pathlib import Path

def process_files(INDIR, MIPPS_DIR, CHUNKSIZE, outdir, MIPPS_BIN, MIPPS_FILE):
    infiles = list(INDIR.glob('*RGB*/*'))
    infiles = [f'"{str(f)}"' for f in infiles] # get posix files

    # setup mipps call
    os.chdir(MIPPS_DIR)

    split = len(infiles) // CHUNKSIZE
    if split == 0: split+=1
    for df in tqdm.tqdm(np.array_split(infiles, split)):
        outlist = ' '.join(df)
        os.makedirs(outdir, exist_ok=True)
        s = f'{MIPPS_BIN} -c={MIPPS_FILE} -o="{outdir}" -j=4 {outlist}'
        # run mipps call
        os.system(s)

CHUNKSIZE = 20
#MIPPS_FILE = r'S:\p_initze\yolov8_object_detection\mipps\33577_all_taps_2018-09-26_13-21-24_modelbased_JPG.mipps'
# 2023
MIPPS_FILE = r'N:\Response\Restricted_Airborne\MACS\Canada\2023_Perma-X_Canada\05_mipps_scripts\MACS_Polar_RGB_mipps\121502_per_pixel_scale_jpg75_rescale.mipps'
MIPPS_DIR = r'C:\Program Files\DLR MACS-Box\bin'
MIPPS_BIN = r'..\tools\MACS\mipps.exe'
OUT_DIR_BASE = Path(r'S:\p_initze\yolov8_object_detection\data')

# file to convert macs images to jpg files
DATA_DIR = Path(r'N:\Response\Restricted_Airborne\MACS\Canada\2023_Perma-X_Canada\1_MACS_original_images')
# List all subdirectories
projects = [x for x in DATA_DIR.iterdir() if (x.is_dir() and x.name.startswith('202'))]

# Print all subdirectories
for project_name in projects:
    print(f'Processing: {project_name}')
    INDIR = DATA_DIR / project_name
    outdir = OUT_DIR_BASE / INDIR.name
    try:
        if not outdir.exists():
                #print(f'Test kickoff {project_name}')
                process_files(INDIR, MIPPS_DIR, CHUNKSIZE, outdir, MIPPS_BIN, MIPPS_FILE)
        else:
            print('Outdir exists, skip processing!')
    except:
        print 
        continue