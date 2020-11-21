
import numpy as np
from numpy.linalg import norm
import skimage.io
import matplotlib.pyplot as plt
from numba import njit


image = skimage.io.imread('map_hsv.png').astype("int")
height, width,_ = image.shape



alpha = 1 
g = alpha * np.array([1,0,0,1]).reshape(2,2)




fi = np.zeros((image[:,:,0].shape[0],image[:,:,0].shape[1],4,2))



# 0 - Vegas gold 
# 1 - Kobicha
c = np.array([[77, 70, 35],[42, 27, 14]])
K = np.array([0,1]) # labels




def get_q(xt,c,kt):
    return int(norm(xt-c[kt]) > norm(xt-c[1-kt]))



def get_neighbours(height,width,i,j):
    # i,j - position of pixel
    n = np.full(4,None)
    inv_neighbours = np.full(4,None)
    inv_inv = np.full(4,None)
    # Left
    if 0<=j-1<width-1 and 0<=i<=height-1:
        n[0]=[i,j-1]
        inv_neighbours[0] = 1
        inv_inv[0] = 0
    # Right
    if 0<j+1<=width-1 and 0<=i<=height-1:
        n[1]=[i,j+1]
        inv_neighbours[1] = 0
        inv_inv[1] = 1
    # Upper
    if 0<=i-1<height-1 and 0<=j<=width-1:
        n[2]=[i-1,j]
        inv_neighbours[2] = 3
        inv_inv[2] = 2
    # Down
    if 0<i+1<=height-1 and 0<=j<=width-1:
        n[3]=[i+1,j]
        inv_neighbours[3] = 2
        inv_inv[3] = 3
    return (list(n[n!=None]),list(inv_neighbours[inv_neighbours!=None]),list(inv_inv[inv_inv!=None]))




for i in range(height):
    for j in range(width):
        neighbours,inv_neighbours,inv_inv = get_neighbours(height, width,i,j)
        len_neighbours = len(neighbours)
        k_ast = np.full(len_neighbours, -100)
        for k in K:
            q = get_q(image[i,j,:],c,k)
            for n in range(len_neighbours):
                n_i,n_j = neighbours[n]
                k_ast[n] = np.argmin(g[k,:]-fi[n_i,n_j,inv_neighbours[n],:])
            k_ast = k_ast[k_ast!=-100] 
            fi_list = [fi[i,j,inv_neighbours[n],k_ast[n] ]  for n,[i,j] in enumerate(neighbours)]
            C_t = (np.sum(g[k,k_ast] - fi_list) + q)/len_neighbours
            fi[i,j,inv_inv,k] = g[k,k_ast]-fi_list-C_t    
                


