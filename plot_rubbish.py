import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def predict(predict_model, input_data):
    pass

def pred_draw(input_file = 'pred_out.npy', output_file = 'true_out.npy', samples = 10, nx = 31, ny = 31):


    try:
        inf = np.load(input_file)
        of = np.load(output_file)
    except:
        print("can't find the file")
        return

    xcord = np.arange(1,nx+1,1)
    ycord = np.arange(1,ny+1,1)
    X,Y = np.meshgrid(xcord,ycord)
    rin = inf.reshape((-1,nx,ny))
    ron = of.reshape((-1,nx,ny))
    for i in range(min(inf.shape[0],samples)):
        fig = plt.figure()
        in_curve = fig.add_subplot('211',projection='3d')
        out_curve = fig.add_subplot('212',projection='3d')
        in_curve.plot_surface(X,Y, rin[i,:,:],cmap='rainbow')
        out_curve.plot_surface(X,Y, ron[i,:,:],cmap='rainbow')
        plt.show()

pred_draw('pred_out.npy', 'true_out.npy')