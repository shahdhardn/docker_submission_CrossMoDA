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
import pathlib
import SimpleITK as sitk
from tqdm.notebook import tqdm

#The following is cloned from the repository 
from hecktor.src.data.utils import read_nifti, write_nifti, get_attributes, resample_sitk_image

#input_dir = '/input/'
#path_img = os.path.join(input_dir,'{}_hrT2.nii.gz')
#path_pred = '/output/{}_Label.nii.gz'


#list_case = [k.split('_hrT2')[0] for k in os.listdir(input_dir)]

main_dir = os.path.join("/docker_submission",'nnUNet/nnunet')
os.environ['nnUNet_raw_data_base'] = os.path.join(main_dir,'nnUNet_raw_data_base')
os.environ['nnUNet_preprocessed'] = os.path.join(main_dir,'preprocessed')
os.environ['RESULTS_FOLDER'] = os.path.join(main_dir,'nnUNet_trained_models')



def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0), out_size = None, 
                                interpolate = sitk.sitkNearestNeighbor):

    # original_spacing = itk_image.GetSpacing()
    # original_size = itk_image.GetSize()

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(interpolate)

    return resample.Execute(itk_image)

INPUT_FOLDER='/ssd_Samsung870_2T/docker_submission/input/'
OUTPUT_FOLDER='/ssd_Samsung870_2T/docker_submission/input_input/'

img_files_v = [] 
org_img_size_v = [] 
org_img_spacing_v = [] 
resampling_size_v = [] 
cropping_size_v = []

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

validation_info = pd.DataFrame(list(zip(img_files_v, org_img_size_v, org_img_spacing_v, 
                                         resampling_size_v, cropping_size_v)), 
                                         columns =['img_files','org_img_size','org_img_spacing',
                                         'resampling_size','cropping_size'])
validation_info.to_csv("validation_info.csv", header=True, index=False)
