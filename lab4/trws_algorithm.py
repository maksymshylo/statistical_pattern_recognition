from numba import njit
import numpy as np

@njit
def get_right_down(height,width,i,j):

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
        N: list
            array of 2 neighbours coordinates
    calculates 2 neighbours: Right and Down
            (i,j)--(i,j+1)
              |
            (i+1,j)
    examples:
    >>> get_right_down(2,2,2,2.)
    Traceback (most recent call last):
    ...
    Exception: invalid indices values (not integer)
    >>> get_right_down(-2,2,2,2)
    Traceback (most recent call last):
    ...
    Exception: height or width is less than zero
    >>> get_right_down(2,2,2,2)
    []
    >>> get_right_down(2,2,0,0)
    [[0, 1], [1, 0]]
    '''
    
    if width <= 0 or height <= 0:
        raise Exception('height or width is less than zero')
    if type(i) is not np.int64 or type(j) is not np.int64:
        raise Exception('invalid indices values (not integer)')

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

    # i,j - position of pixel
    # [Left, Right, Up, Down] - order of possible neighbours
    # array of neighbour indices
    if width <= 0 or height <= 0:
        raise Exception('height or width is less than zero')

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
def forward_pass(height,width,n_labels,Q,g,P,fi):

    '''
    Parameters
        height: int
            height of input image 
        width: int
            width of input image 
        n_labels: int
            number of labels in labelset
        Q: ndarray
            array of unary penalties
        g: ndarray
            array of binary penalties
        P: ndarray
            array consist of best path weight for each direction (Left,Right,Up,Down)
        fi: ndarray
            array of potentials

    Returns
        P: ndarray
            array consist of best path weight for each direction (Left,Right,Up,Down)
        fi: ndarray
            updated array of potentials

    updates fi, according to best path for 'Left' and 'Up' directions
    '''

    # for each pixel of input channel
    for i in range(1,height):
        for j in range(1,width):
            # for each label in pixel
            for k in range(n_labels):
                # P[i,j,0,k] - Left direction
                # P[i,j,2,k] - Up direction
                # calculate best path weight according to formula
                P[i,j,0,k] = max(P[i,j-1,0,:] + (1/2)*Q[i,j-1,:] - fi[i,j-1,:] + g[i,j-1,1,:,k])
                P[i,j,2,k] = max(P[i-1,j,2,:] + (1/2)*Q[i-1,j,:] + fi[i-1,j,:] + g[i-1,j,3,:,k])
                # update potentials
                fi[i,j,k] = (P[i,j,0,k] + P[i,j,1,k] - P[i,j,2,k] - P[i,j,3,k])/2
    return (P,fi)

@njit(fastmath=True, cache=True)
def backward_pass(height,width,n_labels,Q,g,P,fi):

    '''
    Parameters
        height: int
            height of input image 
        width: int
            width of input image 
        n_labels: int
            number of labels in labelset
        Q: ndarray
            array of unary penalties
        g: ndarray
            array of binary penalties
        P: ndarray
            array consist of best path weight for each direction (Left,Right,Up,Down)
        fi: ndarray
            array of potentials

    Returns
        P: ndarray
            array consist of best path weight for each direction (Left,Right,Up,Down)
        fi: ndarray
            updated array of potentials

    updates fi, according to best path for 'Right' and 'Down' directions
    '''

    # for each pixel of input channel
    # going from bottom-right to top-left pixel
    for i in np.arange(height-2,-1,-1):
        for j in np.arange(width-2,-1,-1):
            # for each label in pixel
            for k in range(n_labels):
                # P[i,j,1,k] - Right direction
                # P[i,j,3,k] - Down direction
                # calculate best path weight according to formula
                P[i,j,3,k] = max(P[i+1,j,3,:] + (1/2)*Q[i+1,j,:] + fi[i+1,j,:] + g[i+1,j,2,k,:])
                P[i,j,1,k] = max(P[i,j+1,1,:] + (1/2)*Q[i,j+1,:] - fi[i,j+1,:] + g[i,j+1,0,k,:])
                # update potentials
                fi[i,j,k] = (P[i,j,0,k] + P[i,j,1,k] - P[i,j,2,k] - P[i,j,3,k])/2
    return (P,fi)

def trws(height,width,n_labels,K,Q,g,P,n_iter):

    '''
    Parameters
        height: int
            height of input image 
        width: int
            width of input image 
        n_labels: int
            number of labels in labelset
        K: ndarray
            array of colors (mapping label->color)
        Q: ndarray
            array of unary penalties
        g: ndarray
            array of binary penalties
        P: ndarray
            array consist of best path weight for each direction (Left,Right,Up,Down)
        n_iter: int
            number of iteratations
    Returns
        output: ndarray
            array of optimal labelling (with color mapping)
    one iteration of TRW-S algorithm (forward and backward pass),
    updates fi, according to best path for all directions
    examples:
    >>> trws('height','width','n_labels','K','Q','g','P',-1)
    Traceback (most recent call last):
    ...
    Exception: n_iter <=0
    >>> trws('height','width',3,np.array([0,1]),'Q','g','P',2)
    Traceback (most recent call last):
    ...
    Exception: n_labels do not match with real number of labels
    >>> trws('height','width',2,np.array([0,1]),[],'g','P',2)
    Traceback (most recent call last):
    ...
    Exception: unary or binary penalties are empty
    '''
    if n_iter <= 0:
        raise Exception('n_iter <=0')
    if len(K) != n_labels:
        raise Exception('n_labels do not match with real number of labels')
    if Q==[] or g ==[]:
        raise Exception('unary or binary penalties are empty')

    # initialise array of potentials with zeros
    fi = np.zeros((height,width,n_labels))
    # initialize Right and Down directions
    P,_ = backward_pass(height,width,n_labels,Q,g,P,fi.copy())
    for iteratation in range(n_iter):
        P,fi = forward_pass(height,width,n_labels,Q,g,P,fi)
        P,fi = backward_pass(height,width,n_labels,Q,g,P,fi)
    # restore labelling from optimal energy after n_iter of TRW-S
    labelling = np.argmax(P[:,:,0,:] + P[:,:,1,:] -  fi + Q/2, axis = 2)
    # mapping from labels to colors
    output = K[labelling]
    return output

def optimal_labelling(Q,g,K,n_iter):

    '''
    Parameters
        Q: ndarray
            updated unary penalties
        g: ndarray
            updated binary penalties
        K: ndarray
            set of labels
        n_iter: int
            number of iteratations
    Returns
        labelling: ndarray
        array of optimal labelling (with color mapping) 
        for one channel in input image
    initialize input parameters for TRW-S algorithm,
    and returns best labelling
    '''

    height,width,_ = Q.shape
    n_labels = len(K)
    P = np.zeros((height,width,4,n_labels))
    labelling = trws(height,width,n_labels,K,Q,g,P,n_iter)
    return labelling
