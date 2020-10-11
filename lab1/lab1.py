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
p_k= np.log(p_k, out=np.full_like(p_k,-np.inf), where=(p_k!=0))

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
    #t = len(string)
    ksi = np.random.binomial(size=n*m, n=1, p=noise_level).reshape(image.shape)
    return ksi^image

p = 0.3
string = string_to_image('ok',p)
penalties = np.full([string.shape[1],len(alphabet_list)],-np.inf)
#penalties = np.zeros([string.shape[1],len(alphabet_list)])
#test = np.where(p_k==0, -np.inf, p_k) 

etalones['o'].shape[1],etalones['k'].shape[1]#,etalones['g'].shape[1]


for etalone in etalones:
    ind = etalones[etalone].shape[1]
    if ind <= string.shape[1]:
        s =  string[:,:ind]
        penalties[ind-1,alphabet_dict[etalone]] =             np.sum(np.log(p)*(s^etalones[etalone]) +
                   np.log(1-p)*(1^s^etalones[etalone]) +  p_k[-1,alphabet_dict[etalone]])


proper = np.where((penalties != -np.inf).any(axis=1))[0]


for x in proper:
    for etalone in etalones:
        ind = etalones[etalone].shape[1] + x
        if ind <= string.shape[1]:
            s =  string[:,x:ind]
            penalties[ind-1,alphabet_dict[etalone]] =  max(penalties[ind-1,alphabet_dict[etalone]],                     np.sum(np.log(p)*(s^etalones[etalone]) +
                           np.log(1-p)*(1^s^etalones[etalone])) + 
                                                            p_k[np.argmax(penalties[x-1]),alphabet_dict[etalone]])



np.argmax(penalties[-1]), alphabet_list[np.argmax(penalties[-1])]

np.argmax(penalties[-17]), alphabet_list[np.argmax(penalties[-17])]


'''
#penalties2 = np.full([string.shape[1],len(alphabet_list)],-np.inf)
for x in proper:
    for element in range(len(penalties[x])):
        for etalone in etalones:
            ind = etalones[etalone].shape[1] + x
            if ind <= string.shape[1]:
                s =  string[:,x:ind]
                penalties[ind-1,alphabet_dict[etalone]] =  max(penalties[ind-1,alphabet_dict[etalone]], \
                    (np.sum(np.log(p)*(s^etalones[etalone]) +
                           np.log(1-p)*(1^s^etalones[etalone])) + p_k[element,alphabet_dict[etalone]]) )
'''


