import numpy as np
from numpy import ndarray
from numpy.linalg import norm
import matplotlib.pyplot as plt
from numba import njit
from skimage.io import imread
import sys
import time

# [Left, Right, Up, Down] - order of possible neighbours
# P[i,j,0,k] = L[i,j,k]
# P[i,j,1,k] = R[i,j,k]
# P[i,j,2,k] = U[i,j,k]
# P[i,j,3,k] = D[i,j,k]



@njit(fastmath=True, cache=True)
def forward_pass(height,width,n_labels,Q,g,P,fi):
    # forward
    for i in range(1,height):
        for j in range(1,width):
            for k in range(n_labels):
                P[i,j,0,k] = max(P[i,j-1,0,:] + (1/2)*Q[i,j-1,:] - fi[i,j-1,:] + g[:,k]) # left
                P[i,j,2,k] = max(P[i-1,j,2,:] + (1/2)*Q[i-1,j,:] + fi[i-1,j,:] + g[:,k]) # up
                fi[i,j,k] = (P[i,j,0,k] + P[i,j,1,k] - P[i,j,2,k] - P[i,j,3,k])/2
    return (P,fi)

@njit(fastmath=True, cache=True)
def backward_pass(height,width,n_labels,Q,g,P,fi):
    # backward
    for i in np.arange(height-2,-1,-1):
        for j in np.arange(width-2,-1,-1):
            for k in range(n_labels):
                P[i,j,3,k] = max(P[i+1,j,3,:] + (1/2)*Q[i+1,j,:] + fi[i+1,j,:] + g[k,:]) # down
                P[i,j,1,k] = max(P[i,j+1,1,:] + (1/2)*Q[i,j+1,:] - fi[i,j+1,:] + g[k,:])   # right
                fi[i,j,k] = (P[i,j,0,k] + P[i,j,1,k] - P[i,j,2,k] - P[i,j,3,k])/2
    return (P,fi)

def trws(height,width,n_labels,K,Q,g,P,n_iter):
    fi = np.zeros((height,width,n_labels))
    P,_ = backward_pass(height,width,n_labels,Q,g,P,fi.copy())
    for iteratation in range(n_iter):
        P,fi = forward_pass(height,width,n_labels,Q,g,P,fi)
        P,fi = backward_pass(height,width,n_labels,Q,g,P,fi)
    labelling = np.argmax(P[:,:,0,:] + P[:,:,1,:] -  fi + Q/2, axis = 2)
    output = K[labelling]
    return output


def process_channel(channel,K,alpha,n_iter):
    height, width = channel.shape
    #np.arange(0,255,4)#np.array([0, 64, 128, 192, 255])
    n_labels = len(K)
    P = np.zeros((height,width,4,n_labels)) 
    Epsilon = 0
    #∥xt − kt∥ · 1(xt ̸= Epsilon)
    Q = -abs((channel[:,:,np.newaxis] - K))*(channel[:,:,np.newaxis] != Epsilon)
    g = -alpha*abs(np.subtract(K,K.reshape(-1,1))).astype(float)
    labelling = trws(height,width,n_labels,K,Q,g,P,n_iter)
    return labelling

def process_image(image,K,alpha,n_iter):
    denoised_image = np.zeros_like(image)
    for channel in range(image.shape[2]):
        denoised_image[:,:,channel] = process_channel(image[:,:,channel],K,alpha,n_iter)
    return denoised_image

a = time.time()
image = imread('mona-lisa-damaged.png').astype("int")
alpha = 1
denoised_image = process_image(image,K,alpha,1)
print(time.time()-a)

plt.figure(figsize=(15,15))
plt.axis('off')
plt.imshow(denoised_image)
