import numpy as np
import math

#input/output_file:the name of the input/output files
#samples: number of 'eta's which a file contains
#nx,ny: grid size
#dt: intervals for updating eta
#steps: time steps after initialization
#scale: the limitation of the absolute value of eta
#smooth: times to smooth randomly initialized surface
def random_generator(input_file,output_file,samples=50000,nx=31,ny=31,dt=0.5,steps=10,scale=72,smooth=4):
    #gravity
    gravity = 9.81
    #gird
    Lx = nx - 1
    Ly = ny - 1
    dx = Lx/(nx-1)
    dy = Ly/(ny-1)

    # initialize eta and velocity
    for s in range(samples):
        etap = np.zeros((nx, ny))
        eta = np.zeros((nx, ny))
        etat = np.zeros((nx,ny))
        up = np.zeros((nx, ny))
        u = np.zeros((nx, ny))
        um = np.zeros((nx, ny))
        vp = np.zeros((nx, ny))
        v = np.zeros((nx, ny))
        vm = np.zeros((nx, ny))

        #initialize the surface
        etam = (np.random.random((nx,ny))-0.5)*scale
        #how many times of smoothing
        for times in range(smooth):
            for i in range(1, nx - 1):  # 2:nx-1
                for j in range(1, ny - 1):  # 2:ny-1
                    etat[i][j] = (2*etam[i][j]+etam[i-1][j-1]+etam[i-1][j]+etam[i-1][j+1]+\
                    etam[i][j-1]+etam[i][j+1]+etam[i+1][j-1]+etam[i+1][j]+etam[i+1][j+1])/10
            etam = np.copy(etat)
        eta = np.copy(etam)

        #save the initial surface
        try:
            datai
        except:
            datai = eta.reshape((1, nx, ny, 1))
        else:
            datai = np.concatenate((datai, eta.reshape((1, nx, ny, 1))), axis=0)

        #upadate the eta and velocity
        for n in range(steps):
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    up[i][j] = um[i][j] - gravity * (dt / dx) * (eta[i + 1][j] - eta[i - 1][j])
                    vp[i][j] = vm[i][j] - gravity * (dt / dy) * (eta[i][j + 1] - eta[i][j - 1])
                    etap[i][j] = 2 * eta[i][j] - etam[i][j] + (((2 * math.pow(dt, 2) / dx * dy)) * ( \
                        eta[i + 1][j] + eta[i - 1][j] + eta[i][j + 1] + eta[i][j - 1] - 4 * eta[i][j]))
            etam = np.copy(eta)
            eta = np.copy(etap)

        #save the surface after several steps
        try:
            datao
        except:
            datao = eta.reshape((1, nx, ny, 1))
        else:
            datao = np.concatenate((datao, eta.reshape((1, nx, ny, 1))), axis=0)

    #save the input/output data of samples
    np.save(input_file, datai)
    np.save(output_file, datao)
