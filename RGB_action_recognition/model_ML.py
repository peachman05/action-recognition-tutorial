from keras import Sequential
from keras.layers import CuDNNLSTM
from keras.layers import Dense, TimeDistributed
from keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.python.keras import optimizers
from keras.applications import MobileNetV2
from keras import backend as K

from keras.models import Model

def my_loss( y_true, y_pred ):
    y_pred_softmax = K.softmax(y_pred) 
    return K.sparse_categorical_crossentropy(y_true, y_pred_softmax)

def create_model_pretrain(dim, n_sequence, n_channels, n_output, type='predict'):
    '''
    Paremeter:
        dim -- dimension of image, use (224,224) for MobileNetV2
        n_sequence -- number of sequence(timestep) for LSTM
        n_channels -- number of color channels, use 3 for RGB
        n_output -- number of output class

    '''
    model = Sequential()    
    model.add( 
        TimeDistributed(
            MobileNetV2(weights='imagenet',include_top=False), 
            input_shape=(n_sequence, *dim, n_channels)
        )
    )
    model.add(
        TimeDistributed(
            GlobalAveragePooling2D()
        )
    )
    model.add(CuDNNLSTM(64, return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(.5))

    # Remove softmax in real time usage (use only training process or evaluate)
    if type == 'predict':
        model.add(Dense(n_output))
    else:
        model.add(Dense(n_output, activation='softmax'))
        model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # model.compile(optimizer='sgd', loss=my_loss, metrics=['sparse_categorical_accuracy'])
    
    
    return model

