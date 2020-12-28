import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import sys
import time
from sklearn import mixture
from numba import njit
from trws_algorithm import *
from gmm_algorithm import *
from types import SimpleNamespace



@njit(fastmath=True)
def init_params(image,n_bg,n_fg,bg_params, fg_params,Q):
    height,width,_ = image.shape
    bg_det_cov,bg_weights, bg_inv_cov,bg_means = bg_params
    fg_det_cov,fg_weights, fg_inv_cov,fg_means = fg_params
    c = (2*np.pi)**(-5/2)

    for i in range(height):
        for j in range(width):
            b_prod = np.zeros((n_bg))
            f_prod = np.zeros((n_bg))
            
            for n in range(n_bg):
                b4 = np.exp(-(1/2)*(image[i,j,:]-bg_means[n]).T@bg_inv_cov[n]@(image[i,j,:]-bg_means[n]))
                b_prod[n] = bg_weights[n]*bg_det_cov[n]*b4
                
            for n in range(n_fg):
                f4 = np.exp(-(1/2)*(image[i,j,:]-fg_means[n]).T@fg_inv_cov[n]@(image[i,j,:]-fg_means[n]))            
                f_prod[n] = fg_weights[n]*fg_det_cov[n]*f4
            
            Q[i,j,1] = np.log(np.sum(c*f_prod))
            Q[i,j,0] = np.log(np.sum(c*b_prod))
    return Q




@njit(fastmath=True)
def calculate_beta(image):
    height,width,_ = image.shape
    beta = 0.0
    tau = 2*height*width - height - width
    for i in range(height):
        for j in range(width):
            neighbours = get_right_down(height,width,i,j)
            for [ni,nj] in neighbours:
                beta += np.linalg.norm(image[i,j,:] - image[ni,nj,:])**2
    beta /= tau
    return beta


@njit(fastmath=True)
def calculate_g(image,beta,g):
    height,width,_ = image.shape
    for i in range(height):
        for j in range(width):
            nbs, _ , nbs_indices = get_neighbours(height, width,i,j)
            for n,[n_i,n_j] in enumerate(nbs):
                g[i,j,nbs_indices[n],0,1] = -gamma*np.exp(-np.linalg.norm(image[i,j,:]-image[n_i,n_j,:])**2/(2*beta))
                g[i,j,nbs_indices[n],1,0] = -gamma*np.exp(-np.linalg.norm(image[i,j,:]-image[n_i,n_j,:])**2/(2*beta))
    return g        



def calculate_penalties(image,K,fg,bg,n_fg,n_bg,em_n_iter):

    height, width, _ = image.shape

    fg_means,fg_sigma,_,fg_weights =  gmm(fg, n_fg, n_iter = em_n_iter)
    bg_means,bg_sigma,_,bg_weights =  gmm(bg, n_bg, n_iter = em_n_iter)


    fg_det_cov = np.linalg.det(fg_sigma)**(-1/2)
    fg_inv_cov = np.linalg.inv(fg_sigma)

    bg_det_cov = np.linalg.det(bg_sigma)**(-1/2)
    bg_inv_cov = np.linalg.inv(bg_sigma) 

    bg_params = (bg_det_cov,bg_weights, bg_inv_cov,bg_means)
    fg_params = (fg_det_cov,fg_weights, fg_inv_cov,fg_means)

    Q = np.empty((height,width,len(K)))
    Q =  init_params(image,n_bg,n_fg,bg_params, fg_params,Q)

    beta = calculate_beta(image)

    g = np.zeros((height,width,4,len(K),len(K)))
    g = calculate_g(image,beta,g)

    return Q,g

def process(params):
    params = SimpleNamespace(**params)
    K = np.array([0,1])
    bg = params.image[np.all(params.mask==params.color_bg,axis=2)] # green
    fg = params.image[np.all(params.mask==params.color_fg,axis=2)] # blue
    Q,g = calculate_penalties(params.image, K, fg, bg, params.n_fg, params.n_bg, params.em_n_iter)
    labelling = optimal_labelling(Q,g,K,params.trws_n_iter)

    for _ in range(params.n_iter-1):
        bg = params.image[labelling==0] # green
        fg = params.image[labelling==1] # blue
        Q,g = calculate_penalties(params.image, K, fg, bg, params.n_fg, params.n_bg, params.em_n_iter)
        labelling = optimal_labelling(Q,g,K,params.trws_n_iter)
    return labelling


atime = time.time()

image = imread("lotus.jpg").astype("float64")
mask = imread("lotus-segmentation.png").astype("int")
color_bg = [0,255,0]
color_fg = [0,0,255]
trws_n_iter = 100
em_n_iter = 100
n_iter = 3
gamma = 50
n_bg, n_fg = 3,5

params = {'image': image,
          'mask':mask,
          'color_bg':color_bg,
          'color_fg':color_fg,
          'trws_n_iter':trws_n_iter,
          'em_n_iter':em_n_iter,
          'n_iter':n_iter,
          'gamma':gamma,
          'n_bg':n_bg,
          'n_fg':n_fg
          }


labelling = process(params)
bt = time.time()-atime
print(bt)

plt.figure(figsize = (20,20))

plt.subplot(1,2,1), plt.imshow(image.astype('uint8')), plt.axis('off'), plt.title("input image")

plt.subplot(1,2,2), plt.imshow(labelling), plt.axis('off'),plt.title("segmentation")

plt.show() 

