import json
import time
import numpy as np
import string
import matplotlib.pyplot as plt
from PIL import Image
import os
from numba import njit
import warnings
warnings.filterwarnings('ignore')


def get_bigrams(json_path):
    with open(json_path) as json_file: 
        frequencies_dict = json.load(json_file)
        
    alphabet_list = list(string.ascii_lowercase + ' ')
    alphabet_dict = dict((j,i) for i,j in enumerate(alphabet_list))
    
    array = np.zeros([len(alphabet_list),len(alphabet_list)]).astype('int')
    for i in alphabet_dict:
        for j in alphabet_dict:
            if i + j in frequencies_dict:
                array[alphabet_dict[i]][alphabet_dict[j]] = frequencies_dict[i+j]
                
    p_k = (array.T/array.sum(axis=1)).T 
    p_k= np.log(p_k, out=np.full_like(p_k, -np.inf), where=(p_k!=0))
    
    return (alphabet_list, alphabet_dict, p_k)




def import_images(folder_path,alphabet_list):
    etalones = {}
    for i in alphabet_list[:-1] + ['space']:
        img = np.array(Image.open(folder_path + f'/{i}.png')).astype('int')
        etalones[i] = img
    etalones[' '] = etalones.pop('space')
    return etalones



def string_to_image(string,etalones,noise_level):
    image = etalones[string[0]]
    for i in string[1:]:
        image = np.hstack([image,etalones[i]])
    n,m = image.shape
    if noise_level ==0:
        return image
    else: 
        ksi = np.random.binomial(size=n*m, n=1, p=noise_level).reshape(image.shape)
        return ksi^image


def preprocessing(input_string, alphabet_list, etalones):
    penalties = np.full([input_string.shape[1]+1,len(alphabet_list)],-np.inf)
    letters = list(etalones.values())
    letters_length = [i.shape[1] for i in letters]
    min_letter_size = min([i.shape[1] for i in letters])
    return (penalties, letters,letters_length, min_letter_size)


@njit
def tumba_umba(input_string, letters,penalties, prev_q , next_q):
    for first_el_len in prev_q:
        for i,letter in enumerate(letters):
            cut_till = letter.shape[1] + first_el_len 
            if cut_till <= input_string.shape[1]:
                next_q = np.append(next_q,cut_till)
                x = input_string[:,first_el_len:cut_till]
                f = max(np.sum( (x^letter)*np.log(p) + (1^x^letter)*np.log(1-p) ) + p_k[:,i] 
                            + penalties[first_el_len])
                penalties[cut_till,i] = max(f , penalties[cut_till,i] )
    return (penalties,next_q)


def recognizer(input_string,alphabet_list,etalones,p,p_k):
    
    penalties, letters,letters_length, min_letter_size = preprocessing(input_string, alphabet_list, etalones)
    # first letter
    prev_q = np.array([]).astype("int")
    for i,letter in enumerate(letters):
        cut_till = letter.shape[1]
        if cut_till <= input_string.shape[1]:
            prev_q = np.append(prev_q,cut_till)
            x = input_string[:,:cut_till]
            f = np.sum( (x^letter)*np.log(p) + (1^x^letter)*np.log(1-p)   ) + p_k[-1, i]
            penalties[cut_till,i] = f        
    prev_q = list(set(prev_q))
    while min(prev_q)+min_letter_size <= input_string.shape[1]:
        next_q = np.array([]).astype("int")
        penalties,next_q = tumba_umba(input_string,letters,penalties, prev_q , next_q )
        prev_q = np.array(list(set(next_q)))
        
    output_string = backward_pass(penalties, alphabet_list ,letters_length)
    return output_string


def backward_pass(penalties, alphabet_list ,letters_length):
    pen = np.argmax(penalties,axis=1)[::-1]
    output_string = ''
    i = 0
    while pen.shape[0]-2 >= i:
        last_letter = alphabet_list[pen[i]]
        t = letters_length[pen[i]]
        #print(penalties[-(i+1),pen[i]])
        output_string+=last_letter
        i = i+t
    return output_string[::-1]



a = time.time()
alphabet_list, alphabet_dict, p_k =  get_bigrams('frequencies.json')
etalones = import_images('alphabet',alphabet_list)
p = 0.4
s = 'zdarova otets'
input_string = string_to_image(s,etalones,p) # генеруємо рядок
output_string = recognizer(input_string,alphabet_list,etalones,p,p_k)
print("time", time.time()-a)
print("len:",len(s))
print("input string : ", s)
print("output_string: ", output_string)
