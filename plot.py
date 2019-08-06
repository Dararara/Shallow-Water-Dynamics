import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from keras.models import Model, load_model
from keras.models import Sequential, Input
from keras.layers import Dense, Conv2D, Flatten,Concatenate, MaxPooling2D, Dropout, Conv2DTranspose
from keras import callbacks
import numpy as np
from keras import regularizers
from os import listdir
import re

#model = load_model('best_model_16.h5')
#pred_out = model.predict(x=din.reshape(-1,31,31,1), verbose=1)
#np.save("dpre.npy", pred_out)

din = np.load("random_input.npy")
dout = np.load('random_output.npy')
pred_out = np.load('dpre.npy')

dararara1 = np.zeros(shape = [50000])
dararara2 = np.zeros(shape = [50000])
def inverse_index(test_in, pred_out):
    #get the inverse input and array of predict output
    #return the inverse output data
    index = -1
    mini = 10000000
    keep = 0
    for i in pred_out:
        temp1 = np.sum(np.absolute(i - test_in.reshape(961)))/961
        temp2 = np.sum(np.square(i - test_in.reshape(961)))/961
        dararara1[keep] = temp1
        dararara2[keep] = temp2
        if temp2 < mini:
            mini = temp2
            index = keep
        keep += 1
    np.save('l1.npy', dararara1)
    np.save('l2.npy', dararara2)
    return din[index]




def pred_draw(input_file = 'pred_out.npy', output_file = 'true_out.npy',inf = None, of = None, samples = 10, nx = 31, ny = 31):
    '''
    try:
        inf = np.load(input_file)
        of = np.load(output_file)
    except:
        print("can't find the file")
        return'''
#ri = np.load('D:/project/random/random_in_23333.npy')
#ro = np.load( 'D:/project/random/random_out_23333.npy')

    xcord = np.arange(1,nx+1,1)
    ycord = np.arange(1,ny+1,1)
    X,Y = np.meshgrid(xcord,ycord)
    rin = inf.reshape((-1,nx,ny))
    ron = of.reshape((-1,nx,ny))

    fig = plt.figure(figsize=(32,64))
    i = 0
    in_curve = fig.add_subplot('311',projection='3d')
    out_curve = fig.add_subplot('312',projection='3d')
    in_curve.plot_surface(X,Y, rin[i,:,:],cmap='rainbow')
    out_curve.plot_surface(X,Y, ron[i,:,:],cmap='rainbow')
        
    xmin, xmax = out_curve.get_xlim3d()
    ymin, ymax = out_curve.get_ylim3d()
    #zmin, zmax = out_curve.get_zlim3d()
    zmin,zmax = (-6,6) #Might need to change
    print (xmin,xmax,ymin,ymax,zmin,zmax)
        
    delta = ron[i,:,:]-rin[i,:,:]
    d_curve = fig.add_subplot('313',projection='3d')
        
    d_curve.set(xlim=(xmin,xmax),ylim=(ymin,ymax),zlim=(zmin,zmax))
    d_curve.plot_surface(X,Y,delta,cmap='rainbow')
        #d_curve.view_init(60, 35)
    plt.show()

for i in range(10):
    ran = np.random.randint(0, 50000)
    test_in = pred_out[ran]
    test_out = din[ran]
    result = inverse_index(test_in, pred_out)
    pred_draw(inf=result, of = test_out)
