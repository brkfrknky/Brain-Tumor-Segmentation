# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 20:24:06 2021

@author: a-t-g
"""

import os
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt

from config import config
from train import test_arr,model

def predictByPath(case_path,case):
    files = next(os.walk(case_path))[2]
    X = np.empty((config["VOLUME_SLICES"], config["IMG_SIZE"], config["IMG_SIZE"], 2))
  #  y = np.empty((config["VOLUME_SLICES"], config["IMG_SIZE"], config["IMG_SIZE"]))
    
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_flair.nii.gz');
    flair=nib.load(vol_path).get_fdata()
    
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_t1ce.nii.gz');
    ce=nib.load(vol_path).get_fdata() 
    
 #   vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_seg.nii');
 #   seg=nib.load(vol_path).get_fdata()  

    
    for j in range(config["VOLUME_SLICES"]):
        X[j,:,:,0] = cv2.resize(flair[:,:,j+config["VOLUME_START_AT"]], (config["IMG_SIZE"],config["IMG_SIZE"]))
        X[j,:,:,1] = cv2.resize(ce[:,:,j+config["VOLUME_START_AT"]], (config["IMG_SIZE"],config["IMG_SIZE"]))
 #       y[j,:,:] = cv2.resize(seg[:,:,j+config["VOLUME_START_AT"]], (config["IMG_SIZE"],config["IMG_SIZE"]))
        
  #  model.evaluate(x=X,y=y[:,:,:,0], callbacks= callbacks)
    return model.predict(X/np.max(X), verbose=1)


def showPredictsById(case, start_slice = 60):
    path = config["TRAINING_DIR"] + f"\BraTS20_Training_{case}"
    gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii.gz')).get_fdata()
    origImage = nib.load(os.path.join(path, f'BraTS20_Training_{case}_flair.nii.gz')).get_fdata()
    p = predictByPath(path,case)

    core = p[:,:,:,1]
    edema = p[:,:,:,2]
    enhancing = p[:,:,:,3]

    plt.figure(figsize=(18, 50))
    f, axarr = plt.subplots(1,6, figsize = (18, 50)) 

    axarr[0].imshow(cv2.resize(origImage[:,:,start_slice+config["VOLUME_START_AT"]], (config["IMG_SIZE"], config["IMG_SIZE"])), cmap="gray")
    axarr[0].title.set_text('Original image flair')
    axarr[1].imshow(cv2.resize(gt[:,:,start_slice+config["VOLUME_START_AT"]], (config["IMG_SIZE"], config["IMG_SIZE"])), cmap="gray")
    axarr[1].title.set_text('Ground truth')
    axarr[2].imshow(p[start_slice,:,:,1:4], cmap="gray")
    axarr[2].title.set_text('all classes')
    axarr[3].imshow(edema[start_slice,:,:], cmap="gray")
    axarr[3].title.set_text('NECROTIC predicted')
    axarr[4].imshow(core[start_slice,:,], cmap="gray")
    axarr[4].title.set_text('EDEMA predicted')
    axarr[5].imshow(enhancing[start_slice,:,], cmap="gray")
    axarr[5].title.set_text('ENHANCING predicted')
    plt.show()
    
if __name__ == "__main__":
    showPredictsById(case=test_arr[0][-3:])
    showPredictsById(case=test_arr[1][-3:])
    showPredictsById(case=test_arr[2][-3:])
    showPredictsById(case=test_arr[3][-3:])
    showPredictsById(case=test_arr[4][-3:])
    showPredictsById(case=test_arr[5][-3:])
    showPredictsById(case=test_arr[6][-3:])