from nerual_network import cnn
import numpy as np
din = np.load('random_input.npy')
dout = np.load('random_output.npy')
array = np.array([3,4,5,6,7,8])
array = 2**array
for i in array:
    cnn(din = din, dout = dout, filter_num=i)