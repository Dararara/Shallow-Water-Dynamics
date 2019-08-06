import numpy as np

def inverse_predict(test_in, pred_out, din):
    #get the inverse input and array of predict output
    #return the inverse output data
    index = -1
    mini = 10000000
    keep = 0
    for i in pred_out:
        temp = np.sum(np.square(i - test_in.reshape(961)))/961
        if temp < mini:
            mini = temp
            index = keep
        keep += 1
    return din[index]

