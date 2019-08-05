import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from itertools import cycle, islice
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#g
gravity = 9.81
#网格
Lx = 30
Ly = 30
nx = 31
ny = 31
dx = Lx/(nx-1)
dy = Ly/(ny-1)

flag = 0
title = 1
samples = 50000
limit = 5000

# initialize eta and velocity
for s in range(samples):
    etap = np.zeros((nx, ny))
    eta = np.zeros((nx, ny))
    etam = np.zeros((nx, ny))
    etat = np.zeros((nx,ny))
    up = np.zeros((nx, ny))
    u = np.zeros((nx, ny))
    um = np.zeros((nx, ny))
    vp = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    vm = np.zeros((nx, ny))

    #initialize input
    etam = (np.random.random((nx,ny))-0.5)*72
    for times in range(4):
        for i in range(1, nx - 1):  # 2:nx-1
            for j in range(1, ny - 1):  # 2:ny-1
                etat[i][j] = (2*etam[i][j]+etam[i-1][j-1]+etam[i-1][j]+etam[i-1][j+1]+\
                etam[i][j-1]+etam[i][j+1]+etam[i+1][j-1]+etam[i+1][j]+etam[i+1][j+1])/10
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

    try:
        datai
    except:
        datai = eta.reshape((1, 31, 31, 1))
    else:
        datai = np.concatenate((datai, eta.reshape((1, 31, 31, 1))), axis=0)

    #time
    dt = 0.5
    Nsteps = 10
    for n in range(Nsteps):  # =1:Nsteps:
        t = n * dt
        for i in range(1, nx - 1):  # 2:nx-1
            for j in range(1, ny - 1):  # 2:ny-1
                up[i][j] = um[i][j] - gravity * (dt / dx) * (eta[i + 1][j] - eta[i - 1][j])
                vp[i][j] = vm[i][j] - gravity * (dt / dy) * (eta[i][j + 1] - eta[i][j - 1])
                etap[i][j] = 2 * eta[i][j] - etam[i][j] + (((2 * math.pow(dt, 2) / dx * dy)) * ( \
                    eta[i + 1][j] + eta[i - 1][j] + eta[i][j + 1] + eta[i][j - 1] - 4 * eta[i][j]))
        etam = np.copy(eta)
        eta = np.copy(etap)
    try:
        datao
    except:
        datao = eta.reshape((1, 31, 31, 1))
    else:
        datao = np.concatenate((datao, eta.reshape((1, 31, 31, 1))), axis=0)

np.save('random_input.npy', datai)
np.save('random_output.npy', datao)
