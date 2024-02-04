import os
from os.path import join

import keras
import keras.backend as K
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,CSVLogger

from train_data_csv import split
from Data import DataGenerator
from unet import build_unet

from config import config

######### PARAMETERS ##########
input_layer = Input((config["IMG_SIZE"], config["IMG_SIZE"], 2))
###############################

# PREPARE DATA

train_arr,val_arr,test_arr = split()
training_generator = DataGenerator(train_arr)
valid_generator = DataGenerator(val_arr)
test_generator = DataGenerator(test_arr)

# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_arr)//config["batch_size"]

# CALL MODEL
model = build_unet(input_layer, 'he_normal', 0.2)
model.compile(optimizer= Adam(learning_rate=0.00001),loss='categorical_crossentropy',metrics=['accuracy'])

csv_logger = CSVLogger('training.log', separator=',', append=False)

# DEFINE CALLBACKS
callbacks = [
      keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                               patience=2, verbose=1, mode='auto'),
      keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000001, verbose=1),
      keras.callbacks.ModelCheckpoint(filepath = 'model_.{epoch:02d}-{val_loss:.6f}.m5',
                             verbose=1, save_best_only=True, save_weights_only = True),
        csv_logger
    ]

# TRAINING
def train():
    K.clear_session()
    history =  model.fit(training_generator,
                    epochs=35,
                    steps_per_epoch=steps_per_epoch,
                    callbacks= callbacks,
                    validation_data = valid_generator
                    )  

def save_model(model,model_name):
    if not os.path.exists(config["MODEL_DIR"]):
        os.mkdir(config["MODEL_DIR"])
        
    model_name += ".pt"
    model.save(join(config["MODEL_DIR"],model_name))
    
def load_model(model_dir):
    model = keras.models.load_model(model_dir, compile=False)
    return model

if __name__ == "__main__":
    # start = time_stamp()
    train()
    save_model(model,config["model_file_name"])
    # end = time_stamp()
    # print("training duration: ", end - start)