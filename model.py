from keras import Sequential
from keras.layers import CuDNNLSTM, LSTM
from keras.layers import Dense, Input
from keras.layers import Dropout, concatenate, Flatten
from keras.regularizers import l2
from keras.models import Model

def create_model(num_sequence, num_joints, num_output):
    model = Sequential()
    model.add(CuDNNLSTM(50, input_shape=(num_frame, num_joints),return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(256) )
    model.add(Dropout(0.4))
    model.add(Dense(64) )
    model.add(Dropout(0.4))
    model.add(Dense(num_output, activation='softmax'))
    return model
