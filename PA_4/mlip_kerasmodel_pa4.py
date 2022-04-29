# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 10:50:17 2020

@author: Kather
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
            
x_train, x_test, y_train, y_test = np.asarray(x_train), np.asarray(x_test), np.asarray(y_train), np.asarray(y_test)
#%%
Y_train, Y_test = np.zeros((10000,10),dtype='int'), np.zeros((5000,10),dtype='int')
for i in range(len(y_train)):
    a = y_train[i]
    Y_train[i,a] = 1

for i in range(len(y_test)):
    b = y_test[i]
    Y_test[i,b] = 1 

#%%

from numpy import loadtxt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

model = Sequential()
model.add(Dense(1000, input_dim=784, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(10, activation='sigmoid'))

#%%
# compile the keras model
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#%%

# fit the keras model on the dataset
model.fit(x_train, Y_train, epochs=10, batch_size=100) # training accuracy : 95.69
#%%
# evaluate the keras model
_, accuracy = model.evaluate(x_test, Y_test)
print('Accuracy: %.2f' % (accuracy*100)) # testing accuracy : 96.58                                                                                                                                                                                                                                                                                                 

#%%
# saving the model
import h5py
model.save('mlip_pa4_model.h5')
























