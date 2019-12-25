import numpy as np

from data_gen import DataGenerator
from model_ML import create_model_pretrain
from data_helper import readfile_to_dict

from keras.callbacks.callbacks import Callback
from tensorflow.python.keras.callbacks import ModelCheckpoint

###### Parameters setting
dim = (224,224) # for MobileNetV2
n_sequence = 8 # for LSTM
n_channels = 3 # color channel(RGB)
n_output = 18 # number of output class
batch_size = 2
n_mul_train = 1 # To increase sample of train set
n_mul_test = 4 # To increase sample of test set
path_dataset = 'F:/Master Project/Dataset/KARD-split/'
######

# Keyword argument
params = {'dim': dim,
          'batch_size': batch_size, # you can increase for faster training
          'n_sequence': n_sequence,
          'n_channels': n_channels,
          'path_dataset': path_dataset,
          'option': 'RGBdiff',
          'shuffle': True}

train_txt = "dataset_list/trainlist.txt"
test_txt = "dataset_list/testlist.txt"

# Read file
# train_d and test_d is dictionary that contain name of video as key and label as value
# For example, {'a01\a01_s08_e01': 0, 'a01\a01_s08_e02': 0, .... }
# It's used for getting label(Y)
train_d = readfile_to_dict(train_txt)
test_d = readfile_to_dict(test_txt)

# Prepare key, name of video(X)
train_keys = list(train_d.keys()) * n_mul_train
test_keys = list(test_d.keys()) * n_mul_test

# Generators
training_generator = DataGenerator(train_keys, train_d, **params, type_gen='train')
validation_generator = DataGenerator(test_keys, test_d, **params, type_gen='test')

# Design model
model = create_model_pretrain(dim, n_sequence, n_channels, n_output)
start_epoch = 0

# Load weight of unfinish training model(optional)
load_model = False
if load_model:
    weights_path = 'save_weight/weight-300-0.73-0.81.hdf5' # name of model 
    start_epoch = 300
    model.load_weights(weights_path)

# Set callback
validate_freq = 3
filepath = "save_weight/"+"weight-{epoch:02d}-{accuracy:.2f}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, period=validate_freq)
callbacks_list = [checkpoint]

# # Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=600,
                    callbacks=callbacks_list,
                    initial_epoch=start_epoch,                 
                    validation_freq=validate_freq
                    )

