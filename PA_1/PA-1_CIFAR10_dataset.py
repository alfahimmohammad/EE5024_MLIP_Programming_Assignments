# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 21:46:31 2020

@author: Kather
"""
#task 1 of CIFAr10 dataset
import scipy.io as sio
import scipy
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import stats
# acquiring Xtrain,Xtest,Ytrain and Ytest from the  CIFAR10 dataset
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
# the same way how it's done in task 1 and 2 of MNIST dataset    
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
print('accuracies for every k:',no)
#%%
#task 2 of CIFAR 10 dataset
import scipy.io as sio
import scipy
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import stats

g=np.ones((500,5),dtype='int64')# to store the guessed Y labels for every test case and every K value
for i in range(0,len(Xhist_test[i])):#loopng through every test case
    a=Xhist_test[i]# a single test case
    a=np.reshape(a,(255,1))
    b=np.array(Xhist_train)
    
    det_a=np.sqrt(sum(a**2))[0]# getting the ||a|| value 
    squared_b=b**2
    det_b=np.sqrt(np.sum(squared_b,axis=1,dtype='int64')) # getting the ||b|| value of every sample in the training set
    prod=np.dot(b,a) # getting the numerator value for every training case of NCC
    prod_by_det_a=prod/det_a# dividing the numerator with ||a||
    ncc=np.divide(prod_by_det_a,np.reshape(det_b,(5000,1)))#getting the whole NCC for every 
    #training case by dividing every training case with their respective ||b|| value 
    ncc=ncc.tolist()
    for k in range(0,len(ncc)):
        ncc[k]=ncc[k][0]
    indexes=[]
    for k in range(0,11):# to get the indices of the max values(close to 1) for every K value
        index=ncc.index(max(ncc))
        indexes.append(index)
        ncc[index]=min(ncc)# equating the max value to the min value so that they don't get 
        #get counted in the next iteration
    Ylabel=[]
    for j in range(0,11):# to get the guessed Y labels for every K value
        Ylabel.append(Ytrain[indexes[j]])
    for j in range(0,len(Ylabel)):
        Ylabel[j]=Ylabel[j][0]
    g[i][0]=Ylabel[0]#for k=1
    g[i][1]=stats.mode(Ylabel[0:3])[0].tolist()[0]#for k=3
    g[i][2]=stats.mode(Ylabel[0:5])[0].tolist()[0]#for k=5
    g[i][3]=stats.mode(Ylabel[0:9])[0].tolist()[0]#for k=9
    g[i][4]=stats.mode(Ylabel[0:11])[0].tolist()[0]#for k=11

ytest=np.reshape(Ytest,(500,1))
c=ytest*np.ones((500,5),dtype='int64')
error=g-c
k=[1,3,5,9,11]
k=np.array(k)
no=[]
for i in range(0,5):#looping through every K value
    count=0
    for j in range(0,500):#looping through every test case
        if g[j][i]==c[j][i]:
            count=count+1# counting the no. of correctly guessed test samples
    no.append(count)
no=np.array(no)
no=no/5
plt.plot(k,no)# plotting the accuracy vs k graph
plt.xlabel('k -->')
plt.ylabel('accuracy -->')
plt.title('task 2 using CIFAR10 dataset')
plt.show()
print('the average accuracy :',np.mean(no))
print('accuracies for every k:',no)    