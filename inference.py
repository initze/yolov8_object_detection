#!/usr/bin/env python
# coding: utf-8

from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import numpy as np
from utils import *
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--conf', type=float, default=0.14)
parser.add_argument('--model', type=str, default='models/yolov8l_MACS_beaver_v10best.pt')
args = parser.parse_args()

# setup images
project_name = args.name#'20210703-011838_57_NP_RabbitCreek'
image_dir = Path('data') / project_name
save_dir = Path(f'output/{project_name}')
save_dir_images = save_dir / 'images'
confidence = args.conf#0.14
image_list = list(image_dir.glob('*.jpg'))

model_path = args.model#'models/yolov8l_MACS_beaver_v10best.pt'

def main():
    # setup model
    model = YOLO(model_path)

    # run prediction
    results = model(source=image_dir, conf=confidence, verbose=True)#, save_txt=True, save=True)

    reslist = [get_results(res) for res in results]
    df_output = pd.concat(reslist).reset_index()

    df_class_count = get_class_counts(df_output, image_list=image_list)

    inference_images = df_output['image_path'].unique()

    model.predictor.save_dir = save_dir_images
    for image in inference_images[:]:
        results = model(source=image, conf=confidence, save=True)

    # check if reports dir exists
    if not save_dir.exists():
        os.makedirs()

    # Save outputs
    df_output.to_html(save_dir / 'detected_features.html')
    df_output.to_csv(save_dir / 'detected_features.csv', index=False)

    # save object count summary
    df_class_count.to_html(save_dir / 'detected_image_summary.html')
    df_class_count.to_csv(save_dir / 'detected_image_summary.csv', index=False)

if __name__=='__main__':
    main()