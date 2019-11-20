from keras.layers import Dense, SimpleRNN, Dropout
from keras import Sequential
import numpy as np

n_frames = 2
n_features = 3
X_train = np.array([ [[0,1,2],[3,4,5]],
                     [[1,2,3],[4,5,6]],
                     [[10,11,12],[13,14,15]]
                   ])
Y_train = np.array([6,7,16])

X_test = np.array([ [[2,3,4],[5,6,7]],
                     [[3,4,5],[6,7,8]],
                     [[11,12,13],[14,15,16]]
                   ])
Y_test = np.array([8,9,17])

model = Sequential()
model.add(SimpleRNN(50, input_shape=(n_frames, n_features),return_sequences=False))
# model.add(Dropout(0.3))
model.add(Dense(1) ) # output
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=100)