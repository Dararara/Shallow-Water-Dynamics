from keras.models import Model, load_model
from keras.models import Sequential, Input
from keras.layers import Dense, Conv2D, Flatten,Concatenate, MaxPooling2D, Dropout, Conv2DTranspose
from keras import callbacks
import numpy as np
from keras import regularizers
from os import listdir
import re

def cnn(din , dout , test_din, test_dout, validation_split_num = 0.2,stop_patience = 3, hidden  = 1024, filter_num = 64, kernel_size_1 = (3,3), activation_1 = 'relu'):
    #cnn with one con2D and one hidden layer
    #input is 31 * 31 * 1, output is 961
    #one callback for early stop, one callback to store the best most among all the epochs

    checkpointer = callbacks.ModelCheckpoint(filepath='best_model_{}_{}.h5'.format(filter_num, hidden), verbose=1, save_best_only=True, period=1)
    #tensorboard = callbacks.TensorBoard(write_images=True, log_dir = 'log') 
    mycall = callbacks.EarlyStopping(patience=stop_patience, monitor='loss', mode = 'min', min_delta=0.0001)
    model = Sequential()
    model.add(Conv2D(filters = filter_num, kernel_size = kernel_size_1, activation = activation_1))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(961, activation = 'linear'))
    model.compile(optimizer='Adam', loss='mse', metrics=[ 'mse', 'mae'])
    history = model.fit(batch_size=100, epochs=10000, callbacks=[mycall, checkpointer], x=din.reshape(din.shape[0], 31, 31, 1), y = dout.reshape(dout.shape[0], 961), 
        verbose=2, validation_split = validation_split_num, validation_data=[test_din.reshape(test_din.shape[0], 31, 31, 1), test_dout.reshape(test_dout.shape[0], 961)])
    #print(model.summary())
    #model.save('cnn_{}_{}_{}.h5'.format(filter_num, kernel_size_1[0], kernel_size_1[1]))
    
def test_cnn(din, dout):
    files = listdir()
    model = load_model('best_model_8.h5')
    mse, temp, mae = model.evaluate(x = din.reshape(-1, 31, 31, 1), y = dout.reshape(-1, 961))
    print( mse, mae)
    model = load_model('best_model_16.h5')
    mse, temp, mae = model.evaluate(x = din.reshape(-1, 31, 31, 1), y = dout.reshape(-1, 961))
    print( mse, mae)
    model = load_model('best_model_32.h5')
    mse, temp, mae = model.evaluate(x = din.reshape(-1, 31, 31, 1), y = dout.reshape(-1, 961))
    print( mse, mae)
    model = load_model('best_model_64.h5')
    mse, temp, mae = model.evaluate(x = din.reshape(-1, 31, 31, 1), y = dout.reshape(-1, 961))
    print( mse, mae)
    model = load_model('best_model_128.h5')
    mse, temp, mae = model.evaluate(x = din.reshape(-1, 31, 31, 1), y = dout.reshape(-1, 961))
    print( mse, mae)




            