# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 13:18:33 2021

@author: a-t-g
"""

import matplotlib.pyplot as plt
import os
from os.path import join
import pandas as pd

def draw_loss_graph(hist_path):
    hist = pd.read_csv(hist_path,sep=',',engine='python')
    accuracy = hist['accuracy']
    val_accuracy = hist['val_accuracy']
    
    epoch = range(len(accuracy))
    
    loss = hist['loss']
    val_loss = hist['val_loss']
    
    f,ax=plt.subplots(2,3,figsize=(16,8))
    ax[0,0].plot(epoch,accuracy,'b',label='Training Accuracy')
    ax[0,0].legend()
    
    ax[0,1].plot(epoch,val_accuracy,'r',label='Validation Accuracy')
    ax[0,1].legend()
    
    ax[0,2].plot(epoch,accuracy,'b',label='Training Accuracy')
    ax[0,2].plot(epoch,val_accuracy,'r',label='Validation Accuracy')
    ax[0,2].legend()
    
    ax[1,0].plot(epoch,loss,'b',label='Training Loss')
    ax[1,0].legend()
    
    ax[1,1].plot(epoch,val_loss,'r',label='Validation Loss')
    ax[1,1].legend()
    
    ax[1,2].plot(epoch,loss,'b',label='Training Loss')
    ax[1,2].plot(epoch,val_loss,'r',label='Validation Loss')
    ax[1,2].legend()
    
    plt.show()
    
    
draw_loss_graph('training.log')