import numpy as np
from numpy import ndarray
from numpy.linalg import norm
import skimage.io
import matplotlib.pyplot as plt
from numba import njit
import time

image = skimage.io.imread('map_hsv.png').astype("int")

def get_q(image,c, K):
    norms = np.empty((image.shape[0],image.shape[1],len(K)))
    for k in K:
        norms[...,k] = norm(image-c[k], axis=2)
    
    Q = np.empty_like(norms, dtype = bool)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            Q[i,j,:] = norms[i,j,:] == np.min(norms[i,j,:])
    return Q


@njit
def get_neighbours(height,width,i,j):
    # i,j - position of pixel
    nbs = [] 
    nbs_indices = []
    inv_nbs_indices = []
    # Left
    if 0<=j-1<width-1 and 0<=i<=height-1:
        nbs.append([i,j-1])
        inv_nbs_indices.append(1)
        nbs_indices.append(0)
    # Right
    if 0<j+1<=width-1 and 0<=i<=height-1:
        nbs.append([i,j+1])
        inv_nbs_indices.append(0)
        nbs_indices.append(1)
    # Upper
    if 0<=i-1<height-1 and 0<=j<=width-1:
        nbs.append([i-1,j])
        inv_nbs_indices.append(3)
        nbs_indices.append(2)
    # Down
    if 0<i+1<=height-1 and 0<=j<=width-1:
        nbs.append([i+1,j])
        inv_nbs_indices.append(2)
        nbs_indices.append(3)
    return nbs, inv_nbs_indices, nbs_indices

@njit(fastmath=True, cache=True)
def diff_iter(height: int,width: int, fi: ndarray,g:ndarray,K:ndarray,Q:ndarray) -> ndarray:
    for i in range(height):
        for j in range(width):
            nbs, inv_nbs_indices, nbs_indices = get_neighbours(height, width,i,j)
            len_neighbours = len(nbs)
            k_asterisk = np.full(len_neighbours, -1)
            fi_list = np.full(len_neighbours, np.nan)
            C_t = np.full(len_neighbours, np.nan)
            for k in K:
                for n,[n_i,n_j] in enumerate(nbs):
                    k_asterisk[n] = np.argmax(g[k,:] - fi[n_i,n_j,inv_nbs_indices[n],:])
                    fi_list[n] = fi[n_i,n_j,inv_nbs_indices[n],k_asterisk[n] ]
                    C_t[n] = g[k,k_asterisk[n]] - fi_list[n]
                C_t_sum = (np.sum(C_t) + Q[i,j,k])/len_neighbours
                for n in range(len_neighbours):
                    fi[i,j,nbs_indices[n],k] = C_t[n] - C_t_sum    
    return fi

def diffusion(height,width,K,Q,g, n_iter):
    n_labels = len(K)
    n_neighbors = 4
    fi = np.zeros((height,width,n_neighbors,n_labels))
    for i in range(n_iter):
        fi = diff_iter(height,width,fi,g,K,Q)
    return fi

def get_labelling(height, width, g,fi):
    labelling = np.full((height,width),-1)
    for i in range(height):
        for j in range(width):
            nbs, inv_nbs_indices, nbs_indices = get_neighbours(height, width,i,j)
            n_i, n_j = nbs[0]
            g_reparametrized = g - fi[i,j,nbs_indices[0],:] - fi[n_i,n_j,inv_nbs_indices[0],:]
            labelling[i,j] = np.argmax(np.max(g_reparametrized,axis = 0))
    return labelling

def process(image:ndarray,alpha:int = 1,c:ndarray = np.array([[0,0,255],[0,255,0]]), K:ndarray = np.array([0,1]), n_iter:int = 10 ) -> ndarray:
    height, width, _ = image.shape
    n_labels = len(K)
    n_neighbors = 4
    Q =  get_q(image,c,K)
    g = alpha*np.identity(n_labels)
    fi = diffusion(height, width, K, Q, g, n_iter)
    labelling = get_labelling(height, width, g, fi)
    return labelling

k_arr = process(image,n_iter=10)


