#!/usr/bin/env python
# coding: utf-8



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





p_k = (array.T/array.sum(axis=1)).T # p(k_i+1=h|k_i=p)


etalones = {}
for i in alphabet_list[:-1] + ['space']:
    img = np.array(Image.open("alphabet/{}.png".format(i))).astype('int')
    etalones[i] = img
etalones[' '] = etalones.pop('space')



def string_to_image(string,noise_level):
    image = etalones[string[0]]
    for i in string[1:]:
        image = np.hstack([image,etalones[i]])
    n,m = image.shape
    #t = len(string)
    ksi = np.random.binomial(size=n*m, n=1, p=noise_level).reshape(image.shape)
    return ksi^image




im = string_to_image('aa df fdfs',0)
plt.imshow(im, 'gray')






