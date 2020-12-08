from functionality import *
import matplotlib.pyplot as plt
from skimage.io import imread
import sys
import time



def main():

    '''
    Parameters
        argv[1]: str 
            image_path
        argv[2]: str
            alpha - smoothing coefficient 
        argv[3]: str
            Epsilon - special parameter, which is responsible for lack of color information
        argv[4]: str
            n_labels - number of labels
        argv[5]: str
            n_iter - number of iterations
    '''

    # initialise parameters
    image_path, alpha, Epsilon, n_labels, n_iter = sys.argv[1:6]
    alpha = float(alpha)
    n_iter = int(n_iter)
    Epsilon = int(Epsilon)
    n_labels = int(n_labels)
    
    a = time.time()
    image = imread(image_path).astype("int")
    K = np.arange(0,256,int(256/n_labels))
    denoised_image = process_image(image,K,alpha,Epsilon,n_iter)
    print("total time", time.time()-a)

    plt.figure(figsize = (30,30))

    plt.subplot(1,2,1), plt.imshow(image), plt.axis('off'), plt.title("input image")

    plt.subplot(1,2,2), plt.imshow(denoised_image), plt.axis('off'),plt.title("denoised image")

    plt.show()      



if __name__ == "__main__":
    main()