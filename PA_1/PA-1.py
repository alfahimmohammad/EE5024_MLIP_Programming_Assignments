# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:54:58 2020

@author: Kather
"""

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
    
g=np.ones((500,5),dtype='int32')
#yg=np.ones((500,1),dtype='int32')
for i in range(0,len(Xtest)):
    a=Xtest[i]
    a=np.array(a)
    b=a*np.ones((5000,784),dtype='int32')
    diff=(Xtrain)-b
    squared_diff=diff**2    
    summed=np.sum(squared_diff,axis=1,dtype='int32')
    dist=np.sqrt(summed)
    distance=dist.tolist()
    indices=[]
    for k in range(0,11):
        min_index=distance.index(min(distance))
        indices.append(min_index)
        distance[min_index]=max(distance)+1
    Ylabel=[]
    for j in range(0,11):
        Ylabel.append(Ytrain[indices[j]])
    g[i][0]=Ylabel[0]
    g[i][1]=stats.mode(Ylabel[0:3])[0]
    g[i][2]=stats.mode(Ylabel[0:5])[0]
    g[i][3]=stats.mode(Ylabel[0:9])[0]
    g[i][4]=stats.mode(Ylabel[0:11])[0]
    
ytest=np.reshape(Ytest,(500,1))
c=ytest*np.ones((500,5),dtype='int32')
error=g-c
k=[1,3,5,9,11]
k=np.array(k)
no=[]
for i in range(0,5):
    count=0
    for j in range(0,500):
        if g[j][i]==c[j][i]:
            count=count+1
    no.append(count)
no=np.array(no)
no=no/5
plt.plot(k,no)
plt.xlabel('k -->')
plt.ylabel('accuracy -->')
plt.title('task 1 using MNIST dataset')
plt.show()
print('the average accuracy :',np.mean(no))
#%%
import scipy.io as sio
import scipy
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import stats
Xhist_train=[]
for i in range(0,len(Xtrain)):
    hist=np.histogram(Xtrain[i],bins=[j for j in range(0,256)])[0]
    Xhist_train.append(hist)
Xhist_test=[]
for i in range(0,len(Xtest)):
    hist=np.histogram(Xtest[i],bins=[j for j in range(0,256)])[0]
    Xhist_test.append(hist)
g=np.ones((500,5),dtype='int64')
for i in range(0,len(Xhist_test)):
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
error=g-c
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
#%%
import scipy.io as sio
import scipy
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import stats

Train=sio.loadmat('CIFAR10_Train', mdict=None, appendmat=True)
Test=sio.loadmat('CIFAR10_Test', mdict=None, appendmat=True) 
Xtrain=Train['CIFAR10_Train_Data']
Xtrain=Xtrain.tolist()
Xtest=Test['CIFAR10_Test_Data']
Xtest=Xtest.tolist()
Ytest=Test['CIFAR10_Test_Labels']
Ytest=Ytest.tolist()
Ytrain=Train['CIFAR10_Train_Labels']
Ytrain=Ytrain.tolist()
    
Xhist_train=[]
for i in range(0,len(Xtrain)):
    hist=np.histogram(Xtrain[i],bins=[j for j in range(0,256)])[0]
    Xhist_train.append(hist)
Xhist_test=[]
for i in range(0,len(Xtest)):
    hist=np.histogram(Xtest[i],bins=[j for j in range(0,256)])[0]
    Xhist_test.append(hist)
g=np.ones((500,5),dtype='int64')
for i in range(0,len(Xhist_test)):
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
    g[i][0]=Ylabel[0][0]
    g[i][1]=stats.mode(Ylabel[0:3])[0].tolist()[0][0]
    g[i][2]=stats.mode(Ylabel[0:5])[0].tolist()[0][0]
    g[i][3]=stats.mode(Ylabel[0:9])[0].tolist()[0][0]
    g[i][4]=stats.mode(Ylabel[0:11])[0].tolist()[0][0]
        
ytest=np.reshape(Ytest,(500,1))
c=ytest*np.ones((500,5),dtype='int64')
error=g-c
k=[1,3,5,9,11]
k=np.array(k)
no=[]
for i in range(0,5):
    count=0
    for j in range(0,500):
        if g[j][i]==c[j][i]:
            count=count+1
    no.append(count)
no=np.array(no)
no=no/5
plt.plot(k,no)
plt.xlabel('k -->')
plt.ylabel('accuracy -->')
plt.title('task 1 using CIFAR10 dataset')
plt.show()
print('the average accuracy :',np.mean(no))
#%%
import scipy.io as sio
import scipy
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import stats

g=np.ones((500,5),dtype='int64')
for i in range(0,len(Xhist_test[i])):
    a=Xhist_test[i]
    a=np.reshape(a,(255,1))
    b=np.array(Xhist_train)
    
    det_a=np.sqrt(sum(a**2))[0]
    squared_b=b**2
    det_b=np.sqrt(np.sum(squared_b,axis=1,dtype='int64'))
    prod=np.dot(b,a)
    prod_by_det_a=prod/det_a
    ncc=np.divide(prod_by_det_a,np.reshape(det_b,(5000,1)))
    ncc=ncc.tolist()
    for k in range(0,len(ncc)):
        ncc[k]=ncc[k][0]
    indexes=[]
    for k in range(0,11):
        index=ncc.index(max(ncc))
        indexes.append(index)
        ncc[index]=min(ncc)
    Ylabel=[]
    for j in range(0,11):
        Ylabel.append(Ytrain[indexes[j]])
    for j in range(0,len(Ylabel)):
        Ylabel[j]=Ylabel[j][0]
    g[i][0]=Ylabel[0]
    g[i][1]=stats.mode(Ylabel[0:3])[0].tolist()[0]
    g[i][2]=stats.mode(Ylabel[0:5])[0].tolist()[0]
    g[i][3]=stats.mode(Ylabel[0:9])[0].tolist()[0]
    g[i][4]=stats.mode(Ylabel[0:11])[0].tolist()[0]

ytest=np.reshape(Ytest,(500,1))
c=ytest*np.ones((500,5),dtype='int64')
error=g-c
k=[1,3,5,9,11]
k=np.array(k)
no=[]
for i in range(0,5):
    count=0
    for j in range(0,500):
        if g[j][i]==c[j][i]:
            count=count+1
    no.append(count)
no=np.array(no)
no=no/5
plt.plot(k,no)
plt.xlabel('k -->')
plt.ylabel('accuracy -->')
plt.title('task 2 using CIFAR10 dataset')
plt.show()    
print('the average accuracy :',np.mean(no))