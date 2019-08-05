from keras.models import Model, load_model
from keras.models import Sequential, Input
from keras.layers import Dense, Conv2D, Flatten,Concatenate, MaxPooling2D, Dropout, Conv2DTranspose
from keras import callbacks
import scipy.io as sco
from os import listdir
import re
import numpy as np
import tensorflow as tf
from keras import regularizers
def cnn(din, dout, stop_patience = 3, filter_num = 64, kernel_size_1 = (3,3), activation_1 = 'tanh'):
    
    checkpointer = callbacks.ModelCheckpoint(filepath='best_model.h5', verbose=1, save_best_only=True)
    tensorboard = callbacks.TensorBoard(write_images=True, log_dir = 'log') 
    mycall = callbacks.EarlyStopping(patience=stop_patience, monitor='loss', mode = 'min', min_delta=0.0001)
    model = Sequential()
    model.add(Conv2D(filters = filter_num, kernel_size = kernel_size_1, activation = activation_1))
    model.add(Flatten())
    model.add(Dense(961, activation = 'linear'))
    model.compile(optimizer='Adam', loss='mse', metrics=[ 'mse', 'mae'])
    history = model.fit(batch_size=100, epochs=20, callbacks=[mycall, tensorboard, checkpointer], x=din.reshape(din.shape[0], 31, 31, 1), y = dout.reshape(dout.shape[0], 961), verbose=1)
    model.save('cnn_{}_{}_{}.h5'.format(filter_num, kernel_size_1[0], kernel_size_1[1]))
