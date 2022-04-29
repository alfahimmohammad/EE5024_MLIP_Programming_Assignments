# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 19:05:12 2020

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
# converting Y into one hot vectors
Y_train, Y_test = np.zeros((10000,10),dtype='int'), np.zeros((5000,10),dtype='int')
for i in range(len(y_train)):
    a = y_train[i]
    Y_train[i,a] = 1

for i in range(len(y_test)):
    b = y_test[i]
    Y_test[i,b] = 1 
#%%
# training the model
x0_train = np.ones((10000,1),dtype='float')
x0_test = np.ones((5000,1),dtype='float')
x_train, x_test = np.concatenate((x0_train,x_train),axis=1), np.concatenate((x0_test,x_test),axis=1)
#%%
X_train, X_test = x_train - np.mean(x_train,axis=1), x_test - np.mean(x_test,axis=1)

#%%

def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

def relu(z):
    a, b = np.shape(z)
    for i in range(a):
        for j in range(b):
            if z[i,j] < 0:
                z[i,j] = 0
    return z

def relu_grad(z):
    a, b = np.shape(z)
    for i in range(a):
        for j in range(b):
            if z[i,j] >= 0:
                z[i,j] = 1
            else:
                z[i,j] = 0
    return z

_, m = np.shape(x_train.T)
def forward_n_backward(w1,w2,w3,x,y,alpha,lamda): # w1 - 1000 X 785, w2 - 100 X 1001, w3 - 10 X 101
    _, m = np.shape(x) # m - no. of examples
    b = np.ones((1,m),dtype='float') # 1 X m
    
    #x = np.concatenate((b,x),axis=0) # 785 X m
    a1 = np.tanh(np.dot(w1,x)) # 1000 X m
    a_1 = np.concatenate((b,a1),axis = 0) # 1001 X m
    a2 = np.tanh(np.dot(w2,a_1)) # 100 X m
    a_2 = np.concatenate((b,a2), axis=0) # 101 X m
    h_x = sigmoid(np.dot(w3,a_2)) # 10 X m, sigmoid
    #J = (1/m)*sum(sum(np.square(y-h_x)))
    J = -(1/m)*sum(sum( y * np.log(abs(h_x)) + (1-y) * np.log(abs(1-h_x)) )) #+ (lamda/(2*m))*( sum(sum(np.square(w1[:,1:]))) + sum(sum(w2[:,1:])) )
    
    delta_net3 = (h_x - y) # 10 X m, h_x * (1 - h_x) *
    delta_net2 = (1 - (a_2)**2) * np.dot(w3.T,delta_net3) # 101 X m, (1 - np.square(a_2)), a_2 * (1 - a_2)
    delta_net1 = (1 - (a_1)**2) * np.dot(w2.T,delta_net2[1:,:]) # 1001 X m, (1 - np.square(a_1)), a_1 * (1 - a_1)
    
    #w_3, w_2, w_1 = np.concatenate((np.zeros((np.shape(w3)[0],1), dtype='float'), w3[:,1:]), axis = 1), np.concatenate((np.zeros((np.shape(w2)[0],1), dtype='float'), w2[:,1:]), axis = 1), np.concatenate((np.zeros((np.shape(w1)[0],1), dtype='float'), w1[:,1:]), axis = 1)
    delta_w3 = np.dot(delta_net3,a_2.T)# + (lamda/m) * w_3 # 10 X 101
    delta_w2 = np.dot(delta_net2[1:,:],a_1.T)# + (lamda/m) * w_2 # 100 X 1001
    delta_w1 = np.dot(delta_net1[1:,:],x.T)# + (lamda/m) * w_1 # 1000 X 785
    
    W3 = w3 - alpha * delta_w3 # 10 X 101
    W2 = w2 - alpha * delta_w2 # 100 X 1001
    W1 = w1 - alpha * delta_w1 # 1000 X 785
    
    return J, W1, W2, W3, h_x

np.random.seed(60) # 60

epsilon_init1=np.sqrt(6)/np.sqrt(784+1000)
epsilon_init2=np.sqrt(6)/np.sqrt(1000+100) 
epsilon_init3=np.sqrt(6)/np.sqrt(100+10)    
    
w1 = (np.random.rand(1000,1+784)*2*epsilon_init1)-epsilon_init1    
w2 = (np.random.rand(100,1+1000)*2*epsilon_init2)-epsilon_init2
w3 = (np.random.rand(10,1+100)*2*epsilon_init3)-epsilon_init3    

J_prev = 0
for i in range(200):
    J , w1, w2, w3, h_x = forward_n_backward(w1, w2, w3, x_train.T, Y_train.T, 0.00001, 0 )
    print(i+1,J)
    if i==0:
        J_prev = J
    else:
        if J > J_prev:
            break
        else:
            J_prev = J
W1, W2, W3 = w1, w2, w3 

h = np.reshape(np.amax(h_x,axis=0),(1,m))
H = h
for i in range(9):
    H = np.concatenate((H,h) , axis = 0)

g = np.zeros(np.shape(H),dtype='int')
for j in range(np.shape(H)[1]):
    for i in range(np.shape(H)[0]):
        if H[i,j] == h_x[i,j]:
            g[i,j]=1

count = 0
y =Y_train.T
for i in range(m):
    if list(g[:,i]) == list(y[:,i]):
        count = count + 1
print('accuracy:',count/m*100) # 91.87%
#%%
def forward(w1,w2,w3,x,y):
    _, m = np.shape(x) # m - no. of examples
    b = np.ones((1,m),dtype='float') # 1 X m
    #x = np.concatenate((b,x),axis=0) # 785 X m
    a1 = np.tanh(np.dot(w1,x)) # 1000 X m
    a_1 = np.concatenate((b,a1),axis = 0) # 1001 X m
    a2 = np.tanh(np.dot(w2,a_1)) # 100 X m
    a_2 = np.concatenate((b,a2), axis=0) # 101 X m
    h_x = sigmoid(np.dot(w3,a_2)) # 10 X m, sigmoid
    J = (1/m)*sum(sum(np.square(y-h_x)))
    return J, h_x

J, htest_x = forward(W1, W2, W3, x_test.T, Y_test.T)    

_, n = np.shape(x_test.T)

h = np.reshape(np.amax(htest_x,axis=0),(1,n))
H = h
for i in range(9):
    H = np.concatenate((H,h) , axis = 0)

G = np.zeros(np.shape(H),dtype='int')
for j in range(np.shape(H)[1]):
    for i in range(np.shape(H)[0]):
        if H[i,j] == htest_x[i,j]:
            G[i,j]=1
count = 0
ytest =Y_test.T
for i in range(n):
    if list(G[:,i]) == list(ytest[:,i]):
        count = count + 1
print('accuracy:',count/n*100) # 92.38%

#%%
# five fold cross validation
x, y = x_train.T, Y_train.T
acc = []
for i in range(5):
    Xtest, Ytest = x[:,2000*i:2000*i+2000], y[:,2000*i:2000+2000*i]
    if i==0:
        Xtrain, Ytrain = x[:,2000:10000], y[:,2000:10000]
    elif i==1:
        Xtrain, Ytrain = np.concatenate((x[:,0:2000],x[:,4000:10000]),axis=1), np.concatenate((y[:,0:2000],y[:,4000:10000]),axis=1)
    elif i==2:
        Xtrain, Ytrain = np.concatenate((x[:,0:4000],x[:,6000:10000]),axis=1), np.concatenate((y[:,0:4000],y[:,6000:10000]),axis=1)
    elif i==3:
        Xtrain, Ytrain = np.concatenate((x[:,0:6000],x[:,8000:10000]),axis=1), np.concatenate((y[:,0:6000],y[:,8000:10000]),axis=1)
    else:
        Xtrain, Ytrain = x[:,0:8000], y[:,0:8000]
    
    np.random.seed(60) # 60

    epsilon_init1=np.sqrt(6)/np.sqrt(784+1000)
    epsilon_init2=np.sqrt(6)/np.sqrt(1000+100) 
    epsilon_init3=np.sqrt(6)/np.sqrt(100+10)    
        
    w1 = (np.random.rand(1000,1+784)*2*epsilon_init1)-epsilon_init1    
    w2 = (np.random.rand(100,1+1000)*2*epsilon_init2)-epsilon_init2
    w3 = (np.random.rand(10,1+100)*2*epsilon_init3)-epsilon_init3 
    
    J_prev = 0
    for j in range(200):
        J , w1, w2, w3, h_x = forward_n_backward(w1, w2, w3, Xtrain, Ytrain, 0.00001, 0 )
        print(j+1,J)
        if j==0:
            J_prev = J
        else:
            if J > J_prev:
                break
            else:
                J_prev = J
    W_1, W_2, W_3 = w1, w2, w3 

    J, h_test_x = forward(W_1, W_2, W_3, Xtest, Ytest)  

    _, N = np.shape(Xtest)

    h = np.reshape(np.amax(h_test_x,axis=0),(1,N))
    H = h
    for j in range(9):
        H = np.concatenate((H,h) , axis = 0)
    
    K = np.zeros(np.shape(H),dtype='int')
    for j in range(np.shape(H)[1]):
        for i in range(np.shape(H)[0]):
            if H[i,j] == h_test_x[i,j]:
                K[i,j]=1
    count = 0
    #ytest =Y_test.T
    for j in range(N):
        if list(K[:,j]) == list(Ytest[:,j]):
            count = count + 1
    accuracy = count/N*100
    acc.append(accuracy)

print(np.mean(acc))# 90.25%

#%%
from sklearn.metrics import confusion_matrix
labels = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]) 
confmatrix = confusion_matrix(y_test, g, a) 







