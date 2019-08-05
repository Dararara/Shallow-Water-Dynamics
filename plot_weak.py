import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def pred_draw(mode = 1, inf = None, of = None, input_file='pred_out.npy',output_file='true_out.npy',samples=10,nx=31,ny=31):
    if(mode == 1):
        input_file = input('the input file name: ')
        output_file = input('the output file name: ')
        inf = np.load(input_file)
        of = np.load(output_file)
    if(inf == None or of == None):
        print('data wrong')
        return
#ri = np.load('D:/project/random/random_in_23333.npy')
#ro = np.load( 'D:/project/random/random_out_23333.npy')

    xcord = np.arange(1,nx+1,1)
    ycord = np.arange(1,ny+1,1)
    X,Y = np.meshgrid(xcord,ycord)

    rin = inf.reshape((inf.shape[0],nx,ny))
    ron = of.reshape((of.shape[0],nx,ny))

    for i in range(min(inf.shape[0],samples)):
        fig = plt.figure()
        in_curve = fig.add_subplot('221',projection='3d')
        out_curve = fig.add_subplot('222',projection='3d')
        in_curve.plot_surface(X,Y, rin[i,:,:],cmap='autumn')
        out_curve.plot_surface(X,Y, ron[i,:,:],cmap='autumn')
        delta = ron[i,:,:]-rin[i,:,:]
        d_curve = fig.add_subplot('223',projection='3d')
        d_curve.plot_surface(X,Y,delta,cmap='rainbow')
        plt.show()

#pred_draw('random_output_2.npy','random_pred.npy',10)
pred_draw(2)