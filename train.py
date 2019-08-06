from nerual_network import cnn, test_cnn
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#load data
din = np.load('random_input.npy')
dout = np.load('random_output.npy')
#split data into training data and test data
test_dout = dout[:5000]
test_din = din[:5000]
din = din[5000:]
dout = dout[5000:]

#train cnn with different filter numbers, 8, 16 to 256

cnn(din = din, dout = dout, test_din = test_din, test_dout=test_dout, filter_num = 16, stop_patience=10, hidden=4096)
