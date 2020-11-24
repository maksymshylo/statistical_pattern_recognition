from functionality import *
import sys
import time
from skimage.io import imread
from colordict import ColorDict

def main():

    '''
    Parameters
        argv[1]: str 
            image_path
        argv[2]: str
            alpha - smoothing coefficient 
        argv[3]: int
            n_iter - number of iterations
        argv[4:]: list
            c  - list of colors: black, white etc.
    '''

    # initialise parameters
    image_path, alpha, n_iter = sys.argv[1:4]
    c = sys.argv[4:]
    alpha = float(alpha)
    n_iter = int(n_iter)

    a = time.time()
    image = imread(image_path).astype("int")
    height, width, _ = image.shape
    # colors - dict of rgb colors codes, to see all colors, please use print(ColorDict())
    colors = ColorDict()
    c = np.array([ colors[key] for key in c], dtype = int)
    n_labels = len(c)
    # defining label set
    K = np.arange(n_labels)
    # calculating unary penalties
    Q =  get_q(image,c,K)
    # defining binary penalties as g(x,y) = 1(x=y)
    g = alpha*np.identity(n_labels)
    # calculating potentials
    fi = diffusion(height, width, K, Q, g, n_iter)
    # get optimal labelling
    labelling = get_labelling(height, width, g, c, fi)
    print("total time", time.time()-a)

    plt.figure(figsize = (20,20))

    plt.subplot(1,2,1), plt.imshow(image), plt.axis('off'), plt.title("input image")

    plt.subplot(1,2,2), plt.imshow(labelling), plt.axis('off'),plt.title("denoised image")

    plt.show()      



if __name__ == "__main__":
    main()
