# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 10:48:14 2020

@author: MEENU G H
"""

import pickle
import numpy as np
file_path='mnist.pkl'
with open(file_path, 'rb') as f:
    x = pickle.load(f, encoding='latin1')

y_train=[]
x_train=[]
y_test=[]
x_test=[]
##### generating 1000 per class for training ###########
ij=np.zeros([10,1])
for (ii,i) in enumerate((x[0])[1]):
    for j in range(0,10):
        if ((i==j) and (int(ij[j,0])<1000)):
            x_train.append(((x[0])[0])[ii])
            y_train.append(((x[0])[1])[ii])
            ij[j,0]+=1

##### generating 500 per class for testing ###########            
ik=np.zeros([10,1])
for (ji,k) in enumerate((x[1])[1]):
    for j1 in range(0,10):
        if ((k==j1) and (int(ik[j1,0])<500)):
            x_test.append(((x[0])[0])[ji])
            y_test.append(((x[0])[1])[ji])
            ik[j1,0]+=1