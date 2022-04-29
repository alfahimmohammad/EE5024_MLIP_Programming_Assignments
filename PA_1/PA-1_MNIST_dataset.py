# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 21:30:11 2020

@author: Kather
"""
# task 1 using mnist dataset
import scipy.io as sio
import scipy
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import stats

train=sio.loadmat('mnist_train', mdict=None, appendmat=True)
test=sio.loadmat('mnist_test', mdict=None, appendmat=True)       
a=train['traindata']
Ytrain=[]
Xtrain=[]
for i in range(0,len(a)):
    Ytrain.append(a[i][0])
    x=a[i].tolist()
    del x[0]
    Xtrain.append(x)
b=test['testdata']
Ytest=[]
Xtest=[]
for i in range(0,len(b)):
    Ytest.append(b[i][0])
    x=b[i].tolist()
    del x[0]
    Xtest.append(x)
    
g=np.ones((500,5),dtype='int32') # to store the guessed Y labels for every test case and every K value
for i in range(0,len(Xtest)): # looping through every test case
    a=Xtest[i] # taking one test case
    a=np.array(a)
    b=a*np.ones((5000,784),dtype='int32') # to subtract it from every sample in the training set
    diff=(Xtrain)-b # and subtracting it
    squared_diff=diff**2    # squaring them up
    summed=np.sum(squared_diff,axis=1,dtype='int32') # adding them all in the 'x' direction 
    dist=np.sqrt(summed) # taking square root of every element
    distance=dist.tolist()
    indices=[]
    for k in range(0,11):# getting the index of the most min value and storing them 
        min_index=distance.index(min(distance))
        indices.append(min_index)
        distance[min_index]=max(distance)+1# equating the min value to a high value so that 
        #it doesn't get counted the in the next iteration
    Ylabel=[]
    for j in range(0,11):
        Ylabel.append(Ytrain[indices[j]]) #storing the Ylables of the min values
    g[i][0]=Ylabel[0] # for k=1
    g[i][1]=stats.mode(Ylabel[0:3])[0]# for k=3
    g[i][2]=stats.mode(Ylabel[0:5])[0]# for k=5
    g[i][3]=stats.mode(Ylabel[0:9])[0]# for k=9
    g[i][4]=stats.mode(Ylabel[0:11])[0]#for k=11
    
ytest=np.reshape(Ytest,(500,1))
c=ytest*np.ones((500,5),dtype='int32')
error=g-c
k=[1,3,5,9,11]
k=np.array(k)
no=[]
for i in range(0,5):# looping through every k
    count=0
    for j in range(0,500):# looping through every test case
        if g[j][i]==c[j][i]:
            count=count+1# counting the no. of correctly guessed samples for every k
    no.append(count)
no=np.array(no)
no=no/5
plt.plot(k,no)# plotting the accuracy vs k graph
plt.xlabel('k -->')
plt.ylabel('accuracy -->')
plt.title('task 1 using MNIST dataset')
plt.show()
print('the average accuracy :',np.mean(no))
print('accuracies for every k:',no)
#%% 
#task 2 using mnist dataset
import scipy.io as sio
import scipy
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import stats
# acquiring histogram from datasets for Xtrain and Xtest
Xhist_train=[]
for i in range(0,len(Xtrain)):
    hist=np.histogram(Xtrain[i],bins=[j for j in range(0,256)])[0]
    Xhist_train.append(hist)
Xhist_test=[]
for i in range(0,len(Xtest)):
    hist=np.histogram(Xtest[i],bins=[j for j in range(0,256)])[0]
    Xhist_test.append(hist)
g=np.ones((500,5),dtype='int64') # to store guessed Y labels for every K value and every test case
for i in range(0,len(Xhist_test)): # looping through every sample
    a=Xhist_test[i]
    h=a*np.ones((5000,255),dtype='int64')
    difference=Xhist_train-h
    diff=difference**2
    sqrd_dist=np.sum(diff,axis=1,dtype='int64')
    dist=np.sqrt(sqrd_dist)
    distance=dist.tolist()
    indexes=[]
    for k in range(0,11):
        min_dist=distance.index(min(distance))
        indexes.append(min_dist)
        distance[min_dist]=max(distance)+1
    Ylabel=[]
    for j in range(0,11):
        Ylabel.append(Ytrain[indexes[j]])
    g[i][0]=Ylabel[0]
    g[i][1]=stats.mode(Ylabel[0:3])[0].tolist()[0]
    g[i][2]=stats.mode(Ylabel[0:5])[0].tolist()[0]
    g[i][3]=stats.mode(Ylabel[0:9])[0].tolist()[0]
    g[i][4]=stats.mode(Ylabel[0:11])[0].tolist()[0]
        
ytest=np.reshape(Ytest,(500,1))
c=ytest*np.ones((500,5),dtype='int64')
error=g-c # calculating variations from the actual y values
k=[1,3,5,9,11]
k=np.array(k)
no=[]
for i in range(0,5):
    count=0
    for j in range(0,500):
        if error[j][i]==0:
            count=count+1
    no.append(count)
no=np.array(no)
no=no/5
plt.plot(k,no)
plt.xlabel('k -->')
plt.ylabel('accuracy -->')
plt.title('task 2 using MNIST dataset')
plt.show()
print('the average accuracy :',np.mean(no))
print('accuracies for every k:',no)