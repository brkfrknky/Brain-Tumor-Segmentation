# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 20:46:15 2021

@author: a-t-g
"""

import os

config = dict()

config["VOLUME_SLICES"] = 100
config["VOLUME_START_AT"] = 20
config["IMG_SIZE"] = 128
config["batch_size"] = 1
config["n_channels"] = 2
config["model_file_name"] = "model_1"


######### DIRECTORIES #########
config["SRC_DIR"] = os.getcwd()
config["ROOT_DIR"] = os.path.dirname(os.getcwd())
config["DATA_DIR"] = os.path.join(config["ROOT_DIR"], 'data')
config["TRAINING_DIR"] = os.path.join(config["DATA_DIR"], 'training')
config["TEST_DIR"] = os.path.join(config["DATA_DIR"], 'validation')
config["MODEL_DIR"] = os.path.join(config["DATA_DIR"], 'models')
##############################