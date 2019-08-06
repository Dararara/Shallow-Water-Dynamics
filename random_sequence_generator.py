import numpy as np
import math

#sequence_name: name of the output file
#samples: number of 'eta's which a file contains
#steps: time steps after initialization
#nx,ny: grid (point) size
#dt: intervals for updating eta
def random_rnn_seq(sequence_name='random_sequence.npy',samples=50,nx=31,ny=31,dt=0.5,steps=10):
    # g
    gravity = 9.81
    # grid
    Lx = nx - 1
    Ly = ny - 1
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # initialize eta(perturbation) and velocity
    for s in range(samples):
        etap = np.zeros((nx, ny))
        eta = np.zeros((nx, ny))
        etat = np.zeros((nx, ny))
        up = np.zeros((nx, ny))
        u = np.zeros((nx, ny))
        um = np.zeros((nx, ny))
        vp = np.zeros((nx, ny))
        v = np.zeros((nx, ny))
        vm = np.zeros((nx, ny))

        # initialize water surface
        etam = (np.random.random((nx, ny)) - 0.5) * 72
        #do several times of smoothing
        for times in range(4):
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    etat[i][j] = (2 * etam[i][j] + etam[i - 1][j - 1] + etam[i - 1][j] + etam[i - 1][j + 1] + \
                                  etam[i][j - 1] + etam[i][j + 1] + etam[i + 1][j - 1] + etam[i + 1][j] + etam[i + 1][
                                      j + 1]) / 10
            etam = np.copy(etat)
        '''
        for j in range(1,ny-1):
            etat[0][j] = (etam[0][j-1]+etam[0][j]+etam[0][j+1]+\
            etam[1][j-1]+etam[1][j]+etam[1][j+1])/6
            etat[nx-1][j] = (etam[nx-1][j-1]+etam[nx-1][j]+etam[nx-1][j+1]+\
            etam[nx-2][j-1]+etam[nx-2][j]+etam[nx-2][j+1])/6
        for i in range(1, ny - 1):
            etat[i][0] = (etam[i-1][0]+etam[i][0]+etam[i+1][0]+\
            etam[i-1][1]+etam[i][1]+etam[i+1][1])/6
            etat[i][ny-1] = (etam[i - 1][ny-1] + etam[i][ny-1] + etam[i + 1][ny-1] + \
            etam[i - 1][ny-2] + etam[i][ny-2] + etam[i + 1][ny-2]) / 6
        etat[0][0] = (etam[0][0]+etam[0][1]+etam[1][0]+etam[1][1])/4
        etat[0][ny-1] = (etam[0][ny-1]+etam[0][ny-2]+etam[1][ny-2]+etam[1][ny-1])/4
        etat[nx-1][0] = (etam[nx-1][0]+etam[nx-2][0]+etam[nx-1][1]+etam[nx-2][1])/4
        etat[nx-1][ny-1] = (etam[nx-1][ny-1]+etam[nx-1][ny-2]+etam[nx-2][ny-1]+etam[nx-2][ny-2])/4
        '''

        eta = np.copy(etam)

        #save initial surface
        #dimension 1: no. of the sample
        #dimension 2: time step
        #dimension 3,4: nx,ny
        #dimension 5: channel
        seq = eta.reshape((1, 1, nx, ny, 1))

        #update eta
        for n in range(steps):
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    up[i][j] = um[i][j] - gravity * (dt / dx) * (eta[i + 1][j] - eta[i - 1][j])
                    vp[i][j] = vm[i][j] - gravity * (dt / dy) * (eta[i][j + 1] - eta[i][j - 1])
                    etap[i][j] = 2 * eta[i][j] - etam[i][j] + (((2 * math.pow(dt, 2) / dx * dy)) * (\
                        eta[i + 1][j] + eta[i - 1][j] + eta[i][j + 1] + eta[i][j - 1] - 4 * eta[i][j]))
            etam = np.copy(eta)
            eta = np.copy(etap)
            #save the eta of current time
            seq = np.concatenate((seq,eta.reshape((1,1,31,31,1))), axis=1)

        #save the whole sequence of eta
        try:
            seq_out
        except:
            seq_out = seq
        else:
            seq_out = np.concatenate((seq_out, seq), axis=0)
            print('output shape:'+ str(seq_out.shape))

    np.save(sequence_name, seq_out)

#random_rnn_seq('random_sequence_init.npy',samples=5000,steps=20)
