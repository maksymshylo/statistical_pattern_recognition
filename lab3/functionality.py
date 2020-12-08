import numpy as np
from numba import njit


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
                P[i,j,0,k] = max(P[i,j-1,0,:] + (1/2)*Q[i,j-1,:] - fi[i,j-1,:] + g[:,k])
                P[i,j,2,k] = max(P[i-1,j,2,:] + (1/2)*Q[i-1,j,:] + fi[i-1,j,:] + g[:,k])
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
                P[i,j,3,k] = max(P[i+1,j,3,:] + (1/2)*Q[i+1,j,:] + fi[i+1,j,:] + g[k,:])
                P[i,j,1,k] = max(P[i,j+1,1,:] + (1/2)*Q[i,j+1,:] - fi[i,j+1,:] + g[k,:])
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
    >>> trws('height','width',-5,'K','Q','g','P','n_iter')
    Traceback (most recent call last):
    ...
    Exception: Wrong n_labels parameter
    >>> trws('height','width',5,np.array([-1,566]),'Q','g','P','n_iter')
    Traceback (most recent call last):
    ...
    Exception: Invalid values in K: < 0 or > 255
    >>> trws('height','width',5,np.array([1,34]),'Q',np.identity(3),'P','n_iter')
    Traceback (most recent call last):
    ...
    Exception: Wrong dimensions K or g
    '''

    if type(n_labels) is not int  or n_labels <= 0:
        raise Exception('Wrong n_labels parameter')
    if min(K) < 0 or max(K) > 255:
        raise Exception('Invalid values in K: < 0 or > 255')
    if K.shape[0] != g.shape[0] or K.shape[0] != g.shape[1]:
        raise Exception('Wrong dimensions K or g')
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

def process_channel(channel,K,alpha,Epsilon,n_iter):

    '''
    Parameters
        channel: int
            index of channel to process 
        K: ndarray
            array of colors (mapping label->color)
        alpha: float
            smoothing coefficient
        Epsilon: int
            special parameter, which is responsible for lack of color information
        n_iter: int
            number of iteratations

    Returns
        labelling: ndarray
            array of optimal labelling (with color mapping) 
            for one channel in input image

    initialize input parameters for TRW-S algorithm,
    and returns best labelling for given channel
    examples:
    >>> process_channel('channel','K','alpha',-1,'n_iter')
    Traceback (most recent call last):
    ...
    Exception: Wrong Epsilon value: < 0 or > 255
    >>> process_channel('channel','K',-1,1,'n_iter')
    Traceback (most recent call last):
    ...
    Exception: Wrong alpha value: < 0
    '''

    if not 0<Epsilon<255:
        raise Exception('Wrong Epsilon value: < 0 or > 255')
    if alpha < 0:
        raise Exception('Wrong alpha value: < 0')

    height, width = channel.shape
    n_labels = len(K)
    P = np.zeros((height,width,4,n_labels))
    # |xt − kt| · 1(xt != Epsilon)
    # calculate unary and binary penalties
    Q = -abs((channel[:,:,np.newaxis] - K))*(channel[:,:,np.newaxis] != Epsilon)
    g = -alpha*abs(np.subtract(K,K.reshape(-1,1))).astype(float)
    labelling = trws(height,width,n_labels,K,Q,g,P,n_iter)
    return labelling

def process_image(image,K,alpha,Epsilon,n_iter):

    '''
    Parameters
        image: ndarray
            input image, which we want to denoise 
        K: ndarray
            width of input image
        alpha: float
            smoothing coefficient
        Epsilon: int
            special parameter, which is responsible for lack of color information
        n_iter: int
            number of iteratations

    Returns
        denoised_image: ndarray
            denoised image after 'n_iter' iteratations of TRW-S

    process all channels in input image
    examples:
    >>> process_image('image','K','alpha','Epsilon','n_iter')
    Traceback (most recent call last):
    ...
    Exception: Wrong n_iter parameter
    >>> process_image(np.array([1]),'K','alpha','Epsilon',1)
    Traceback (most recent call last):
    ...
    Exception: image is empty
    '''

    if type(n_iter) is not int  or n_iter <= 0:
        raise Exception('Wrong n_iter parameter')

    if image.shape[0] <= 1 or image.shape[1] <= 1:
        raise  Exception('image is empty')

    denoised_image = np.zeros_like(image)
    for channel in range(image.shape[2]):
        denoised_image[:,:,channel] = process_channel(image[:,:,channel],K,alpha,Epsilon,n_iter)
    return denoised_image

