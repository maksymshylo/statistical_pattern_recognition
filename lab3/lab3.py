
import numpy as np
from numpy import ndarray
from numpy.linalg import norm
import matplotlib.pyplot as plt
from numba import njit
#from skimage.color import rgba2rgb, rgb2gray
from skimage.io import imread
import sys
import time


image = imread('mona-lisa-damaged.png').astype("int")
image = image[:,:,0]
height, width = image.shape
print(image.shape)
K = np.array([0, 64, 128, 192, 255])
n_labels = len(K)


# [Left, Right, Up, Down] - order of possible neighbours



P = np.zeros((height,width,4,n_labels)) 
Epsilon = 0
#∥xt − kt∥ · 1(xt ̸= Epsilon)
Q = abs((image[:,:,np.newaxis] - K))*(image[:,:,np.newaxis] != Epsilon)
g = abs(np.subtract(K,K.reshape(-1,1)))

for i in reversed(range(height-1)):
    for j in reversed(range(width-1)):
        for k in range(n_labels):
            P[i,j,3,k] = max(P[i+1,j,3,:] + (1/2)*Q[i+1,j,:] + g[k,:]) # down
            P[i,j,1,k] = max(P[i,j+1,1,:] + (1/2)*Q[i,j,:] + g[k,:])   # right


fi = np.zeros((height,width,n_labels))
for i in range(1,height):
    for j in range(1,width):
        for k in range(n_labels):
            P[i,j,0,k] = max(P[i,j-1,0,:] + (1/2)*Q[i,j-1,:] - fi[i,j-1,:] + g[:,k]) # left
            P[i,j,2,k] = max(P[i-1,j,2,:] + (1/2)*Q[i-1,j,:] + fi[i-1,j,:] + g[:,k]) # up
            fi[i,j,k] = (P[i,j,0,k] + P[i,j,1,k] - P[i,j,2,k] - P[i,j,3,k])/2


labelling = np.empty((height,width,len(c[0])),dtype = int)
for i in range(height):
    for j in range(width):
        nbs, inv_nbs_indices, nbs_indices = get_neighbours(height, width,i,j)
        # take any neighbour
        n_i, n_j = nbs[0]
        # calculating reparametrized binary penalties
        g_reparametrized = g - fi[i,j,nbs_indices[0],:] - fi[n_i,n_j,inv_nbs_indices[0],:]
        # g - is supermodular so take the highest possible maximum edge between nodes t, t'
        labelling[i,j,:] = c[np.argmax(np.max(g_reparametrized,axis = 0))]





