#adapted from solution by Paul Bacher
#https://www.kaggle.com/code/paulbacher/custom-preprocessor-rsna-breast-cancer

"""
# Install required libraries
!pip install -qU python-gdcm pydicom pylibjpeg
!pip install -qU dicomsdl
"""

import os
import time

#outside datasets, may need to be installed
import numpy as np
import pandas as pd
import cv2
import pydicom
import dicomsdl

#progress bar (pip install tqdm)
#like used in final project
from tqdm.notebook import tqdm, trange

#data visualization, can choose which one to use
import matplotlib.pyplot as plt
#import seaborn as sns #pip install seaborn


csv_path = '/kaggle/input/rsna-breast-cancer-detection/train.csv'
train_path = '/kaggle/input/rsna-breast-cancer-detection/train_images'
data = pd.read_csv(csv_path)


"""
The following methods we probably won't have to change.
The only thing that might need to be changed is the path to the csv file.
Meanwhile, we should focus on adding our own implemenation to the class,
and building on the preprocessing steps provdied by Paul.
"""

"""
Returns a list of n paths to dicom files and a list of n dictionaries 
mapping patient_id to scan_id. If shuffle is True, the paths are shuffled.
"""
def get_paths(n: int=len(data), shuffle: bool=False):
    if shuffle == True:
        df = data.sample(frac=1, random_state=0)
    else:
        df = data
    paths = []
    ids_cache = []
    for i in range(n):
        patient = str(df.iloc[i]['patient_id'])
        scan = str(df.iloc[i]['image_id'])
        paths.append(train_path + '/' + patient + '/' + scan + '.dcm')
        ids_cache.append({'patient_id': patient, 'scan_id': scan})
    return paths, ids_cache

#per paul's explanation, an optional preprocessor can be added
#to preprocess the images before calculating the aspect ratios

"""
Returns a list of aspect rations for the images in paths.
If a preprocessor is provided, the images are preprocessed before.
"""
def calculate_aspect_ratios(paths: list, preprocessor=None):
    ratios = []
    for i in trange(len(paths)):
        if preprocessor:
            img = preprocessor.preprocess_single_image(paths[i])
        else:
            scan = pydicom.dcmread(paths[i])
            img = scan.pixel_array
        height, width = img.shape
        ratio = height / width
        ratios.append(ratio)
    return ratios

