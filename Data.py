# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:21:45 2021

@author: a-t-g
"""

import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
import os
import cv2
import nilearn as nl
import nibabel as nib
import nilearn.plotting as nlplt

from config import config

TRAINING_DIR = config["TRAINING_DIR"]


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, dim=(config["IMG_SIZE"],config["IMG_SIZE"]), batch_size = config["batch_size"], n_channels = config["n_channels"], shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size*config["VOLUME_SLICES"], *self.dim, self.n_channels))
        y = np.zeros((self.batch_size*config["VOLUME_SLICES"], 240, 240))
        Y = np.zeros((self.batch_size*config["VOLUME_SLICES"], *self.dim, 4))

        
        # Generate data
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(TRAINING_DIR, i)
            case_path = case_path + '/'

            data_path = os.path.join(case_path, f'{i}_flair.nii.gz');
            flair = nib.load(data_path).get_fdata()    

            data_path = os.path.join(case_path, f'{i}_t2.nii.gz');
            ce = nib.load(data_path).get_fdata()
            
            data_path = os.path.join(case_path, f'{i}_seg.nii.gz');
            seg = nib.load(data_path).get_fdata()
        
            for j in range(config["VOLUME_SLICES"]):
                 X[j +config["VOLUME_SLICES"]*c,:,:,0] = cv2.resize(flair[:,:,j+config["VOLUME_START_AT"]], (config["IMG_SIZE"], config["IMG_SIZE"]));
                 X[j +config["VOLUME_SLICES"]*c,:,:,1] = cv2.resize(ce[:,:,j+config["VOLUME_START_AT"]], (config["IMG_SIZE"], config["IMG_SIZE"]));

                 y[j +config["VOLUME_SLICES"]*c] = seg[:,:,j+config["VOLUME_START_AT"]];
                    
        # Generate masks
        y[y==4] = 3;
        mask = tf.one_hot(y, 4);
        Y = tf.image.resize(mask, (config["IMG_SIZE"], config["IMG_SIZE"]));
        return X/np.max(X), Y