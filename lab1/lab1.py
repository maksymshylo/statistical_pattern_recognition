import json
import time
import numpy as np
import string
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
from numba import njit
import warnings
warnings.filterwarnings('ignore')

def get_bigrams(json_path):


    '''
    Parameters
        json_path: str
            path to json file with frequencies
    Returns
        alphabet_list: list
            list of letters
        alphabet_dict: dict
            dict of letters with index(a:0)
        p_k: ndarray
            array with shape (letters x letters)

    loads pair letters frequencies
    ''' 
    with open(json_path) as json_file: 
        frequencies_dict = json.load(json_file)

    # create alphabet
    alphabet_list = list(string.ascii_lowercase + ' ')
    alphabet_dict = dict((j,i) for i,j in enumerate(alphabet_list))

    # all pairs as array
    array = np.zeros([len(alphabet_list),len(alphabet_list)]).astype('int')
    for i in alphabet_dict:
        for j in alphabet_dict:
            if i + j in frequencies_dict:
                array[alphabet_dict[i]][alphabet_dict[j]] = frequencies_dict[i+j]

    # make a-priopi probabilities from frequencies
    p_k = (array.T/array.sum(axis=1)).T

    # make -np.inf where p(k) = 0
    p_k = np.log(p_k, out=np.full_like(p_k, -np.inf), where=(p_k!=0))

    return (alphabet_list, alphabet_dict, p_k)

def import_images(folder_path,alphabet_list):
    
    '''
    Parameters
        folder_path: str
            path to folder with images
        alphabet_list: list
            list of letters
    Returns
        reference_images: dict
            dict (key - letter, values - ndarray of letter)

    importing images in dict
    '''

    reference_images = {}
    for i in alphabet_list[:-1] + ['space']:
        img = np.array(Image.open(folder_path + f'/{i}.png')).astype('int')
        reference_images[i] = img
    reference_images[' '] = reference_images.pop('space')
    return reference_images


def string_to_image(string,reference_images,noise_level):
    
    '''
    Parameters
        string: str
            path to folder with images
        reference_images: dict
            dict (key - letter, values - ndarray of letter)
        noise_level: float
            level of noising

    Returns
        output_image: ndarray
            noised string as array

    generating image from string with noising
    '''

    # create string as array
    image = reference_images[string[0]]
    for i in string[1:]:
        image = np.hstack([image,reference_images[i]])
    n,m = image.shape

    # generate binomial noise
    ksi = np.random.binomial(size=n*m, n=1, p=noise_level).reshape(image.shape)
    output_image = ksi^image
    return output_image

def preprocessing(input_image, alphabet_list, reference_images):

    '''
    Parameters
        input_image: ndarray
            noised image
        alphabet_list: list
            list of letters
        reference_images: dict
            dict (key - letter, values - ndarray of letter)
    Returns
        penalties: ndarray
            array with shape (input_image +1, alphabet_list) filled -inf
        letters: list
            list of reference_images arrays
        letters_length: list
            list of letters lengths
        min_letter_size: int
            minimum letter width

    initializating of parameters for recognizing process
    '''
    # initializing penalties
    penalties = np.full([input_image.shape[1]+1,len(alphabet_list)],-np.inf)
    letters = list(reference_images.values())
    letters_length = [i.shape[1] for i in letters]
    min_letter_size = min([i.shape[1] for i in letters])
    return (penalties, letters,letters_length, min_letter_size)

@njit
def tumba_umba(input_image, letters, penalties, prev_q , next_q, p, p_k):
    
    '''
    Parameters
        input_image: ndarray
            noised image
        letters: list
            list of reference_images arrays
        penalties: ndarray
            array with shape (input_image +1, alphabet_list) filled -inf
        prev_q: ndarray
            array of indices where previous letter ends
        next_q: ndarray
            array of indices where next letter ends
        p: float
            noise level
        p_k: ndarray
            array with shape (letters x letters)
    Returns
        penalties: ndarray
            array with shape (input_image +1, alphabet_list) filled -inf
        next_q: ndarray
            array of indices where next letter ends

    calculating penalties for pair of letters
    '''
    # check all previous lengths
    for prev_len in prev_q:

        # trying to match with reference letter
        for i,letter in enumerate(letters):

            # index of next letter lenght
            cut_till = letter.shape[1] + prev_len 
            if cut_till <= input_image.shape[1]:
                next_q = np.append(next_q,cut_till)

                # matching reference letter with slice of input image
                x = input_image[:,prev_len:cut_till]

                # calculate penalties
                f = max(np.sum( (x^letter)*np.log(p) + (1^x^letter)*np.log(1-p) ) + p_k[:,i] 
                            + penalties[prev_len])
                penalties[cut_till,i] = max(f , penalties[cut_till,i] )
    return (penalties,next_q)

def recognizer(input_image,alphabet_list,reference_images,p,p_k):

    '''
    Parameters
        alphabet_list: list
            list of letters
        reference_images: dict
            dict (key - letter, values - ndarray of letter)
        p: float
            noise level
        p_k: ndarray
            array with shape (letters x letters)
    Returns
        output_image: ndarray
            recognized image

    recognizing noised image
    '''

    if p == 0:
        return input_image
    if p == 1:
        return 1^input_image

    # initialize parameters
    penalties, letters, letters_length, min_letter_size = preprocessing(input_image, alphabet_list, reference_images)
    # first letter
    prev_q = np.array([]).astype("int")

    # recognize first letter
    for i,letter in enumerate(letters):
        cut_till = letter.shape[1]
        if cut_till <= input_image.shape[1]:
            prev_q = np.append(prev_q,cut_till)
            x = input_image[:,:cut_till]
            f = np.sum( (x^letter)*np.log(p) + (1^x^letter)*np.log(1-p)   ) + p_k[-1, i]
            penalties[cut_till,i] = f        
    prev_q = list(set(prev_q))

    # 2, etc. 
    # until we reach end of string
    while min(prev_q) + min_letter_size <= input_image.shape[1]:
        next_q = np.array([]).astype("int")
        penalties,next_q = tumba_umba(input_image,letters,penalties, prev_q , next_q, p, p_k)
        prev_q = np.array(list(set(next_q)))
    
    # create string from penalties array
    output_string = backward_pass(penalties, alphabet_list ,letters_length)
    output_image = string_to_image(output_string,reference_images,0)
    return output_image

def backward_pass(penalties, alphabet_list ,letters_length):

    '''
    Parameters
        penalties: ndarray
            filled with penalties
        alphabet_list: list
            list of letters
        letters_length: list
            list of letters lengths
    Returns
        output_string[::-1]: str
            recognized string

    get recognized string from penalties
    '''
    # best letters in each length
    pen = np.argmax(penalties,axis=1)[::-1]
    output_string = ''
    i = 0
    # until we reach end of string
    while pen.shape[0]-2 >= i:
        last_letter = alphabet_list[pen[i]]
        t = letters_length[pen[i]]
        #print(penalties[-(i+1),pen[i]])
        output_string+=last_letter
        i = i+t
    return output_string[::-1]





def main():
    '''
    Parameters
        argv[1]: str 
            path to json
        argv[2]: str
            path to image folder
        argv[3]: string
            input string
        argv[4]: float
            noise level
    '''
    # initialise parameters
    path_to_json, image_folder, input_string, p = sys.argv[1:]
    a = time.time()
    p = float(p)
    alphabet_list, alphabet_dict, p_k =  get_bigrams(path_to_json)
    reference_images = import_images(image_folder,alphabet_list)
    input_image_zero_noise = string_to_image(input_string,reference_images,0)
    input_image = string_to_image(input_string,reference_images,p)

    # recognizing string 
    output_image = recognizer(input_image,alphabet_list,reference_images,p,p_k)
    print("total time", time.time()-a)
    print("length:",len(input_string))
    print("input string : ", input_string)

    # plot images
    plt.figure()

    plt.subplot(3,1,1), plt.imshow(input_image_zero_noise,cmap = 'gray'), plt.axis('off')

    plt.subplot(3,1,2), plt.imshow(input_image,cmap = 'gray'), plt.axis('off')

    plt.subplot(3,1,3), plt.imshow(output_image,cmap = 'gray'), plt.axis('off')

    plt.show()    


if __name__ == "__main__":
    main()
