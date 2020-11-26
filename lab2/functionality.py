import numpy as np
from numpy import ndarray
from numpy.linalg import norm
import matplotlib.pyplot as plt
from numba import njit

def get_q(image,c, K):

    '''
    Parameters
        image: ndarray
            input image as array
        c: ndarray
            color mapping
        K: ndarray
            label set
    Returns
        Q: ndarray
            unary penalties for image

    calculates unary penalties for input image, accordingly to c-mapping

    examples:
    >>> get_q('image',np.array([[0,0,0]]), 'K' )
    Traceback (most recent call last):
    ...
    Exception: less than two colors

    >>> get_q('image',np.array([[0,0,0],[255,255,0]]), np.array([0,3]) )
    Traceback (most recent call last):
    ...
    Exception: incorrect set of labels
    '''

    if c.shape[0] < 2:
        raise Exception('less than two colors')
    if (K != np.arange(c.shape[0])).any():
        raise Exception('incorrect set of labels')
    
    # calculating norms for every label
    norms = np.empty((image.shape[0],image.shape[1],len(K)))
    for k in K:
        norms[...,k] = norm(image-c[k], axis=2)
    mins = np.argmin(norms,axis=(2))
    # unary penalties
    Q = np.zeros_like(norms, dtype = bool)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # q(k) = 1(x_t = min(norm(C(k) - X_t)))
            Q[i,j,mins[i,j]] = True
    return Q


@njit
def get_neighbours(height,width,i,j):

    '''
    Parameters
        height: int
            height of input image 
        width: int
            width of input image 
        i: int
            number of row
        j: int
            number of column
    Returns
        N: tuple
            neighbours coordinates, neighbours indices, inverse neighbours indices

    calculates neighbours in 4-neighbours system (and inverse indices)

            (i-1,j)
               |
    (i,j-1)--(i,j)--(i,j+1)
               |
            (i+1,j)

    examples:

    >>> get_neighbours(-1,1,1,1)
    Traceback (most recent call last):
    ...
    Exception: height or width is less than zero
    >>> get_neighbours(3,3,-1,-1)
    ([], [], [])
    >>> get_neighbours(3,3,0,0)
    ([[0, 1], [1, 0]], [0, 2], [1, 3])
    >>> get_neighbours(3,3,0,1)
    ([[0, 0], [0, 2], [1, 1]], [1, 0, 2], [0, 1, 3])
    >>> get_neighbours(3,3,1,1)
    ([[1, 0], [1, 2], [0, 1], [2, 1]], [1, 0, 3, 2], [0, 1, 2, 3])
    '''

    if width <= 0 or height <= 0:
        raise Exception('height or width is less than zero')

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

@njit(fastmath=True, cache=True)
def diff_iter(height: int, width: int, fi: ndarray, g:ndarray, K:ndarray, Q:ndarray) -> ndarray:

    '''
    Parameters
        height: int
            height of input image 
        width: int
            width of input image 
        fi: ndarray
            array of potentials
        g: ndarray
            array of binary penalties
        K: ndarray
            label set
        Q: ndarray
            array of unary penalties
    Returns
        fi: ndarray
            updated array of potentials

    calculates one iteration of diffusion 
    '''
    # for each pixel of input image
    for i in range(height):
        for j in range(width):
            # defining neighbours pixel (i,j)
            nbs, inv_nbs_indices, nbs_indices = get_neighbours(height, width,i,j)
            len_neighbours = len(nbs)
            k_asterisk = np.full(len_neighbours, -1)
            fi_list = np.full(len_neighbours, np.nan)
            C_t = np.full(len_neighbours, np.nan)
            # for each label in label set
            for k in K:
                # for each neighbour of pixel (i,j)
                for n,[n_i,n_j] in enumerate(nbs):
                    # k*(t`)
                    k_asterisk[n] = np.argmax(g[k,:] - fi[n_i,n_j,inv_nbs_indices[n],:])
                    # fi(t',t)(k*)
                    fi_list[n] = fi[n_i,n_j,inv_nbs_indices[n],k_asterisk[n] ]
                    C_t[n] = g[k,k_asterisk[n]] - fi_list[n]
                C_t_sum = (np.sum(C_t) + Q[i,j,k])/len_neighbours
                # updating potentials
                for n in range(len_neighbours):
                    fi[i,j,nbs_indices[n],k] = C_t[n] - C_t_sum    
    return fi

def diffusion(height,width,K,Q,g, n_iter):

    '''
    Parameters
        height: int
            height of input image 
        width: int
            width of input image 
        K: ndarray
            label set
        Q: ndarray
            array of unary penalties
        g: ndarray
            array of binary penalties
        n_iter: int
            number of iteration
    Returns
        fi: ndarray
            updated array of potentials after n iterations

    n iterations of diffusion
    examples:
    >>> diffusion('height','width','K','Q','g', 0)
    Traceback (most recent call last):
    ...
    Exception: number of iterations is less or equal to zero
    '''

    if n_iter <= 0:
        raise Exception('number of iterations is less or equal to zero')

    n_labels = len(K)
    # 4 system neighbourhood
    n_neighbors = 4
    # initialize potentials as zeros
    fi = np.zeros((height,width,n_neighbors,n_labels))
    for i in range(n_iter):
        fi = diff_iter(height,width,fi,g,K,Q)
    return fi

def get_labelling(height, width, g, c, fi):

    '''
    Parameters
        height: int
            height of input image 
        width: int
            width of input image 
        g: ndarray
            array of binary penalties
        c: ndarray
            color mapping
        fi: ndarray
            updated array of potentials
    Returns
        labelling: ndarray
            optimal labels after diffusion algorithm

    labelling restoration after diffusion
    '''

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
    return labelling


if __name__ == "__main__":
    import doctest
    doctest.testmod()



