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

#load data
din = np.load('random_input.npy')
dout = np.load('random_output.npy')
#split data into training data and test data
test_dout = dout[:dout.shape[0]/10]
test_din = din[:din.shape[0]/10]
din = din[din.shape[0]/10:]
dout = dout[dout.shape[0]/10:]

model = load_model('best_model_16.h5')
r = np.random.randint(high = din.shape[0], low = 0, size= 10)#here we choose 10 samples from dataset
pout = model.predict(x = din[r].reshape(10, 31, 31, 1))
pout = pout.reshape(10, 31,31)
print(dout[r].shape, pout.shape)
np.save('pred_out.npy', pout)
np.save('true_out.npy', dout[r])