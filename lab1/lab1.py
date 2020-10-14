import json
import numpy as np
import string
import matplotlib.pyplot as plt
from PIL import Image
import os



with open('frequencies.json') as json_file: 
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



etalones = {}
for i in alphabet_list[:-1] + ['space']:
    img = np.array(Image.open(f'alphabet/{i}.png')).astype('int')
    etalones[i] = img
etalones[' '] = etalones.pop('space')




def string_to_image(string,noise_level):
    image = etalones[string[0]]
    for i in string[1:]:
        image = np.hstack([image,etalones[i]])
    n,m = image.shape
    if noise_level ==0:
        return image
    else: 
        ksi = np.random.binomial(size=n*m, n=1, p=noise_level).reshape(image.shape)
        return ksi^image



p = 0.4
string = string_to_image('my',p) # генеруємо рядок
# генеруємо масив з -np.inf шейпом (ширина рядка х алфавіт)
penalties = np.full([string.shape[1]+1,len(alphabet_list)],-np.inf)
letters = list(etalones.values())

#etalones['o'].shape[1],etalones['f'].shape[1]

# first letter
q = []
for i,letter in enumerate(letters):
    cut_till = letter.shape[1]
    if cut_till <= string.shape[1]:
        q.append(cut_till)
        x = string[:,:cut_till]
        f = np.sum( (x^letter)*np.log(p) + (1^x^letter)*np.log(1-p)   ) + p_k[-1, i]
        penalties[cut_till,i] = f
        
q = list(set(q))


# second letter
for first_el_len in q:
    
    for index_f, f_first_el in enumerate(penalties[first_el_len]):
        
        if f_first_el != -np.inf:
            
            for i,letter in enumerate(letters):
                cut_till = letter.shape[1] + first_el_len 
                if cut_till <= string.shape[1]:
                    x = string[:,first_el_len:cut_till]
                    f = np.sum( (x^letter)*np.log(p) + (1^x^letter)*np.log(1-p) ) + p_k[index_f, i] + f_first_el
                    penalties[cut_till,i] = max(f , penalties[cut_till,i] )




last_argmax = np.argmax(penalties[-1])
last_letter = alphabet_list[last_argmax]
print(last_argmax,last_letter)




index_f = penalties.shape[0]-etalones[last_letter].shape[1]-1
first_argmax =  np.argmax(penalties[index_f])
print(first_argmax, alphabet_list[first_argmax] )






