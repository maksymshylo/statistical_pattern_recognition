#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
from numpy.linalg import norm
import skimage.io
import matplotlib.pyplot as plt
from numba import njit


image = skimage.io.imread('map_hsv.png').astype("int")
height, width,_ = image.shape



alpha = 1 
g = alpha * np.array([1,0,0,1]).reshape(2,2)



fi = np.zeros((height,width,4,2))



# 0 - Vegas gold 
# 1 - Kobicha
c = np.array([[0,0,255],[0,255,0]])
K = np.array([0,1]) # labels



def get_q(xt,c,k):
    return norm(xt-c[k], axis=2) > norm(xt-c[1-k], axis=2)  



Q1 = np.full((height,width,2), np.nan)
for k in K:
    Q1[:,:,k] = get_q(image,c,k)




@njit
def get_neighbours(height,width,i,j):
    # i,j - position of pixel
    n = [] #np.full((4,2),np.nan)
    inv_neighbours = []
    inv_inv = []
    # Left
    if 0<=j-1<width-1 and 0<=i<=height-1:
        n.append([i,j-1])
        inv_neighbours.append(1)
        inv_inv.append(0)
    # Right
    if 0<j+1<=width-1 and 0<=i<=height-1:
        n.append([i,j+1])
        inv_neighbours.append(0)
        inv_inv.append(1)
    # Upper
    if 0<=i-1<height-1 and 0<=j<=width-1:
        n.append([i-1,j])
        inv_neighbours.append(3)
        inv_inv.append(2)
    # Down
    if 0<i+1<=height-1 and 0<=j<=width-1:
        n.append([i+1,j])
        inv_neighbours.append(2)
        inv_inv.append(3)
    return n, inv_neighbours,inv_inv


get_neighbours(height,width,0,2)



import time



a = time.time()
for it in range(10):
    for i in range(height):
        for j in range(width):
            neighbours,inv_neighbours,inv_inv = get_neighbours(height, width,i,j)
            len_neighbours = len(neighbours)
            k_ast = np.full((2,len_neighbours), np.nan)
            
            q = Q1[i,j,:]
            for n in range(len_neighbours):
                n_i,n_j = neighbours[n]
                #k_ast[0,n] = np.argmax(g[0,:]-fi[n_i,n_j,inv_neighbours[n],:])
                #k_ast[1,n] = np.argmax(g[1,:]-fi[n_i,n_j,inv_neighbours[n],:])
                k_ast[:,n] = np.argmax((g-fi[n_i,n_j,inv_neighbours[n],:]),axis=1)
            k_ast = k_ast.astype('int')
            #k_ast = k_ast[k_ast!=[-100,-100]]
            #print(k_ast,i,j)
            fi_list = [[fi[i,j,inv_neighbours[n],k_ast[0,n] ]  for n,[i,j] in enumerate(neighbours)],
                       [fi[i,j,inv_neighbours[n],k_ast[1,n] ]  for n,[i,j] in enumerate(neighbours)]]

            C_t = [(np.sum(g[0,k_ast[0,:]] - fi_list[0]) + q[0])/len_neighbours,
                      (np.sum(g[1,k_ast[1,:]] - fi_list[1]) + q[1])/len_neighbours]

            fi[i,j,inv_inv,0] = g[0,k_ast[0,:]]-fi_list[0]-C_t[0]    
            fi[i,j,inv_inv,1] = g[1,k_ast[1,:]]-fi_list[1]-C_t[1]
            
            #fi[i,j,inv_inv,:] = g[:,k_ast]-fi_list-C_t
print(time.time()-a)



k_arr = np.full((height,width),np.inf)




for i in range(height):
    for j in range(width):
        neighbours,inv_neighbours,inv_inv = get_neighbours(height, width,i,j)
        n_i,n_j = neighbours[0]
        g_fi = g-fi[i,j,inv_inv[0],:]-fi[n_i,n_j,inv_neighbours[0],:]
        k_arr[i,j] = np.argmin(np.min(g_fi,axis = 0))




r = np.full((height,width,3),np.inf)
for i in range(height):
    for j in range(width):
        r[i,j,:]=c[int(k_arr[i,j])]
r = r.astype('uint8')



plt.figure(figsize=(20,20))
plt.imshow(r)




