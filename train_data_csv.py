# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 08:08:54 2021

@author: a-t-g
"""
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import numpy as np

from config import config

def split():  
    survival_info_df = pd.read_csv(os.path.join(config["TRAINING_DIR"],'survival_info.csv'))
    name_mapping_df = pd.read_csv(os.path.join(config["TRAINING_DIR"],'name_mapping.csv'))
    name_mapping_df.rename({'BraTS_2020_subject_ID': 'Brats20ID'}, axis=1, inplace=True) 
    df = survival_info_df.merge(name_mapping_df, on="Brats20ID", how="right")
    paths = []
    
    for _, row  in df.iterrows():
        
        id_ = row['Brats20ID']
        phase = id_.split("_")[-2]
        
        if phase == 'Training':
            path = os.path.join(config["TRAINING_DIR"], id_)
        else:
            path = os.path.join(config["TRAINING_DIR"], id_)
        paths.append(path)
        
    df['path'] = paths
    
    train_data = df.loc[df['Age'].notnull()].reset_index(drop=True)
    train_data["Age_rank"] =  train_data["Age"] // 10 * 10
    train_data = train_data.loc[train_data['Brats20ID'] != 'BraTS20_Training_355'].reset_index(drop=True, )
    
    skf = StratifiedKFold(
        n_splits=5, random_state=55, shuffle=True
    )
    for i, (train_index, val_index) in enumerate(
            skf.split(train_data, train_data["Age_rank"])
            ):
            train_data.loc[val_index, "fold"] = i
    
    test_df = df.loc[~df['Age'].notnull()].reset_index(drop=True)
    train_df = train_data.loc[train_data['fold'] != 0].reset_index(drop=True)
    val_df = train_data.loc[train_data['fold'] == 0].reset_index(drop=True)
    train_data.to_csv("train_data.csv", index=False)
    
    train_arr = train_df['Brats20ID'].to_numpy()
    val_arr = val_df['Brats20ID'].to_numpy()
    test_arr = test_df['Brats20ID'].to_numpy()
    
    return train_arr,val_arr,test_arr
