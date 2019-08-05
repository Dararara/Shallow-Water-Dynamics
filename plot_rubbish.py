import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def pred_draw(input_file,output_file,samples):
    inf = np.load(input_file)
    of = np.load(output_file)

#ri = np.load('D:/project/random/random_in_23333.npy')
#ro = np.load( 'D:/project/random/random_out_23333.npy')

    Lx = 30
    Ly = 30
    nx = 31
    ny = 31
    dx = Lx/(nx-1)
    dy = Ly/(ny-1)
    xcord = np.arange(1,nx+1,1)
    ycord = np.arange(1,ny+1,1)
    X,Y = np.meshgrid(xcord,ycord)

    rin = inf.reshape((inf.shape[0],nx,ny))
    ron = of.reshape((of.shape[0],nx,ny))

    for i in range(min(inf.shape[0],samples)):
        fig = plt.figure()
        in_curve = fig.add_subplot('211',projection='3d')
        out_curve = fig.add_subplot('212',projection='3d')
        in_curve.plot_surface(X,Y, rin[i,:,:],cmap='rainbow')
        out_curve.plot_surface(X,Y, ron[i,:,:],cmap='rainbow')
        plt.show()
