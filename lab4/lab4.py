import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import sys
import time
from trws import *
from sklearn import mixture
from numba import njit

atime = time.time()
image = imread("lotus.jpg").astype("float64")
height, width, _ = image.shape
mask = imread("lotus-segmentation.png").astype("int")


bg = image[np.all(mask==[0,255,0],axis=2)] # green
fg = image[np.all(mask==[0,0,255],axis=2)] # blue
gamma = 50


n_bg, n_fg = 3,3
em_bg = mixture.GaussianMixture(n_components=n_bg)
em_bg.fit(bg)
em_fg = mixture.GaussianMixture(n_components=n_fg)
em_fg.fit(fg)


@njit
def get_right_down(height,width,i,j):
    # i,j - position of pixel
    # [Right, Down] - order of possible neighbours
    # array of neighbour indices
    nbs = [] 
    # Right
    if 0<j+1<=width-1 and 0<=i<=height-1:
        nbs.append([i,j+1])
    # Down
    if 0<i+1<=height-1 and 0<=j<=width-1:
        nbs.append([i+1,j])
    return nbs

@njit
def get_neighbours(height,width,i,j):
    # i,j - position of pixel
    # [Left, Right, Up, Down] - order of possible neighbours
    # array of neighbour indices
    nbs = [] 
    # neighbour indices
    nbs_indices = []
    # inverse neighbour indices
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
    N = (nbs, inv_nbs_indices, nbs_indices)
    return N

c = (2*np.pi)**(-5/2)

fg_det_cov = np.linalg.det(em_fg.covariances_)**(-1/2)
fg_weights = em_fg.weights_.copy()
fg_inv_cov = np.linalg.inv(em_fg.covariances_)
fg_means = em_fg.means_.copy()

bg_det_cov = np.linalg.det(em_bg.covariances_)**(-1/2)
bg_weights = em_bg.weights_.copy()
bg_inv_cov = np.linalg.inv(em_bg.covariances_) 
bg_means = em_bg.means_.copy()

bg_params = (bg_det_cov,bg_weights, bg_inv_cov,bg_means)
fg_params = (fg_det_cov,fg_weights, fg_inv_cov,fg_means)
Q = np.empty((height,width,2))

@njit(fastmath=True)
def init_params(image,height,width,n_bg,bg_params, fg_params,Q):
    bg_det_cov,bg_weights, bg_inv_cov,bg_means = bg_params
    fg_det_cov,fg_weights, fg_inv_cov,fg_means = fg_params
    
    for i in range(height):
        for j in range(width):
            b_prod = np.zeros((n_bg))
            f_prod = np.zeros((n_bg))
            for n in range(n_bg):

                b4 = np.exp(-(1/2)*(image[i,j,:]-bg_means[n]).T@bg_inv_cov[n]@(image[i,j,:]-bg_means[n]))
                b_prod[n] = c*bg_weights[n]*bg_det_cov[n]*b4
            
                f4 = np.exp(-(1/2)*(image[i,j,:]-fg_means[n]).T@fg_inv_cov[n]@(image[i,j,:]-fg_means[n]))            
                f_prod[n] = c*fg_weights[n]*fg_det_cov[n]*f4
            
            Q[i,j,1] = np.log(np.sum(f_prod))
            Q[i,j,0] = np.log(np.sum(b_prod))
    return Q



a = time.time()
Q =  init_params(image,height,width,n_bg,bg_params, fg_params,Q)
print(time.time()-a)


@njit(fastmath=True)
def calculate_beta(height, width, image):
    beta = 0.0
    tau = 0
    for i in range(height):
        for j in range(width):
            neighbours = get_right_down(height,width,i,j)
            for [ni,nj] in neighbours:
                beta += np.linalg.norm(image[i,j,:] - image[ni,nj,:])**2
                tau += 1
    beta /= tau
    return beta

a = time.time()
beta = calculate_beta(height, width, image)
print(time.time()-a)


g = np.zeros((height,width,4,2,2))


@njit(fastmath=True)
def calculate_g(image,height,width,beta,g):
    for i in range(height):
        for j in range(width):
            nbs, _ , nbs_indices = get_neighbours(height, width,i,j)
            for n,[n_i,n_j] in enumerate(nbs):
                g[i,j,nbs_indices[n],0,1] = -gamma*np.exp(-np.linalg.norm(image[i,j,:]-image[n_i,n_j,:])**2/(2*beta))
                g[i,j,nbs_indices[n],1,0] = -gamma*np.exp(-np.linalg.norm(image[i,j,:]-image[n_i,n_j,:])**2/(2*beta))
    return g        

a = time.time()
g = calculate_g(image,height,width,beta,g)
print(time.time()-a)


def process(K,n_iter):
    n_labels = len(K)
    P = np.zeros((height,width,4,n_labels))
    labelling = trws(height,width,n_labels,K,Q,g,P,n_iter)
    return labelling

labelling = process(np.array([0,1]),10)

plt.imshow(labelling)


bt = time.time()-atime