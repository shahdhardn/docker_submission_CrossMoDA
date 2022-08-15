#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from xml.etree.ElementPath import prepare_predicate
import monai
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import ImageGrid
import skimage.io as io
import os
from skimage.transform import resize
import torchio as tio
import pandas as pd 
import sys
import json
import pathlib
import SimpleITK as sitk
from tqdm.notebook import tqdm


#input_dir = '/input/'
#path_img = os.path.join(input_dir,'{}_hrT2.nii.gz')
#path_pred = '/output/{}_Label.nii.gz'


#list_case = [k.split('_hrT2')[0] for k in os.listdir(input_dir)]

main_dir = os.path.join("/docker_submission",'nnUNet/nnunet')
os.environ['nnUNet_raw_data_base'] = os.path.join(main_dir,'nnUNet_raw_data_base')
os.environ['nnUNet_preprocessed'] = os.path.join(main_dir,'preprocessed')
os.environ['RESULTS_FOLDER'] = os.path.join(main_dir,'nnUNet_trained_models')

INPUT_FOLDER='/ssd_Samsung870_2T/docker_submission/input/'
OUTPUT_FOLDER='/ssd_Samsung870_2T/docker_submission/input_input/'

img_files_v = [] 
org_img_size_v = [] 
org_img_spacing_v = [] 
resampling_size_v = [] 
cropping_size_v = []

#Pre-process
for i in os.listdir(INPUT_FOLDER): 
    
    img_files_v.append(i)

    image = tio.ScalarImage(INPUT_FOLDER+i)
    
    org_img_size_v.append(list(image.shape[1:]))
    org_img_spacing_v.append(image.spacing)

    transform_resampling = tio.Resample((1, 1, 1), image_interpolation="bspline")
    image_resampled = transform_resampling(image)

    resampling_size_v.append(list(image_resampled.shape[1:]))

    target_size = [256, 256, image_resampled.shape[3]]
    transform_cropad = tio.CropOrPad(target_size)
    image_cropped = transform_cropad(image_resampled)

    cropping_size_v.append(list(image_cropped.shape[1:]))

    image_cropped.save(OUTPUT_FOLDER+i[:-7]+'_0000'+'.nii.gz')

test_df = pd.DataFrame(list(zip(img_files_v, org_img_size_v, org_img_spacing_v, 
                                         resampling_size_v, cropping_size_v)), 
                                         columns =['img_files','org_img_size','org_img_spacing',
                                         'resampling_size','cropping_size'])
test_df.to_csv("test_info.csv", header=True, index=False)

#Check the sizes for confirmation
for i in os.listdir(OUTPUT_FOLDER):

    image = tio.ScalarImage(os.path.join(OUTPUT_FOLDER, i))
    label_size = list(image.shape[1:3])    

    if label_size == [256, 256]: 
        continue 
    else: 
        transform = tio.CropOrPad([256, 256])
        label_transform = transform(image)
        label_transform.save(os.path.join(OUTPUT_FOLDER,i))
  
'''Run model here'''

main_dir = os.path.join("/ssd_Samsung870_2T",'nnUNet/nnunet')
os.environ['nnUNet_raw_data_base'] = os.path.join(main_dir,'nnUNet_raw_data_base')
os.environ['nnUNet_preprocessed'] = os.path.join(main_dir,'preprocessed')
os.environ['RESULTS_FOLDER'] = os.path.join(main_dir,'nnUNet_trained_models')

os.system('nnUNet_find_best_configuration -m 3d_fullres -tr nnUNetTrainerV2 nnUNetTrainerV2_insaneDA nnUNetTrainerV2_Loss_DiceCE_noSmooth -t 119')

os.system('nnUNet_predict -i /ssd_Samsung870_2T/docker_submission/input/ -o /ssd_Samsung870_2T/docker_submission/output_output_hou/ -t 119 -tr nnUNetTrainerV2 -m 3d_fullres --save_npz')
os.system('nnUNet_predict -i /ssd_Samsung870_2T/docker_submission/input/ -o /ssd_Samsung870_2T/docker_submission/output_output_shahad/ -t 119 -tr nnUNetTrainerV2_Loss_DiceCE_noSmooth -m 3d_fullres --save_npz')
os.system('nnUNet_predict -i /ssd_Samsung870_2T/docker_submission/input/ -o /ssd_Samsung870_2T/docker_submission/output_output_hussain/ -t 119 -tr nnUNetTrainerV2_insaneDA -m 3d_fullres --save_npz')

os.system('nnUNet_ensemble -f /ssd_Samsung870_2T/docker_submission/output_output_hou/ /ssd_Samsung870_2T/docker_submission/output_output_shahad/ /ssd_Samsung870_2T/docker_submission/output_output_hussain/ -o /ssd_Samsung870_2T/docker_submission/output_output/ -pp POSTPROCESSING_FILE(json) ')
'''Post Processing'''

INPUT_FOLDER = "/ssd_Samsung870_2T/docker_submission/output_output" #fix here
ORI_PATH = "/ssd_Samsung870_2T/docker_submission/input"
OUTPUT_FOLDER = "/ssd_Samsung870_2T/docker_submission/output" #fix here
os.makedirs(OUTPUT_FOLDER,exist_ok=True)

test_df = pd.read_csv("/ssd_Samsung870_2T/docker_submission/test_info.csv") #fix here

for i in os.listdir(INPUT_FOLDER): 

    k = i[:-7] +"_hrT2.nii.gz" #how it was saved in the test csv 
    #In the csv file, the sizes are saved as a string of list, so we need to convert it to list
    list_str = test_df.loc[test_df['img_files'] == k, 'resampling_size'].item() #pick the size for the img 
    target_size = json.loads(list_str)

    label = tio.LabelMap(os.path.join(INPUT_FOLDER,i))

    transform_1 = tio.CropOrPad(target_size)
    label_transform_1 = transform_1(label)

    #In the csv file, the spacings are saved as a string of tuple, so we need to convert it to tuple
    spacing_str = test_df.loc[test_df['img_files'] == k, 'org_img_spacing'].item() #pick the spacing for the img 
    target_spacing = eval(spacing_str)

    transform_2 = tio.Resample(target_spacing, image_interpolation='nearest', label_interpolation="nearest")
    label_transform_2 = transform_2(label_transform_1)

    s = i[:-7] + "_Label.nii.gz"
    label_transform_2.save(os.path.join(OUTPUT_FOLDER,s))


#Confirm label sizes match original test image sizes
for i in os.listdir(OUTPUT_FOLDER):

    label = tio.LabelMap(os.path.join(OUTPUT_FOLDER, i))
    label_size = list(label.shape[1:4])

    k = i[:21] +"_hrT2.nii.gz" #Remove "_Label.nii.gz" and add "_hrT2.nii.gz" for the testing images
    
    #In the csv file, the sizes are saved as a string of list, so we need to convert it to list
    list_str = test_df.loc[test_df['img_files'] == k, 'org_img_size'].item() #pick the size for the img 
    org_size = json.loads(list_str)

    if label_size == org_size: 
        continue 
    else: 
        transform = tio.CropOrPad(org_size)
        label_transform = transform(label)
        label_transform.save(os.path.join(OUTPUT_FOLDER,i))

