import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def pred_draw(inf = None, of = None, nx = 31, ny = 31):
    # draw the 3d plot for two input matrices, with the error difference plot
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

