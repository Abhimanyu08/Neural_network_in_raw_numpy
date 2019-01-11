#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import genfromtxt
import math
import matplotlib.pyplot as plt
import sys


# In[2]:


train_data = genfromtxt('mnist_train.csv', delimiter = ',')


# In[3]:


test_data = genfromtxt('mnist_test.csv',delimiter = ',')


# In[4]:


def data_cleaner(dataset):
    dataset = dataset[1:,:].T
    X = dataset[1:, :]
    Y = dataset[0, :]
    return X,Y


# In[5]:


X_train,Y_train = data_cleaner(train_data)
print(X_train.shape)
print(Y_train.shape)


# In[6]:


X_test,Y_test = data_cleaner(test_data)
print(X_test.shape)
print(Y_test.shape)


# In[7]:


def Y_to_softmax(Y):
    yt = np.zeros((int(np.max(Y))+1,Y.shape[0]))
    for i in range(len(Y)):
        yt[int(Y[i]),int(i)] = 1.0
    return yt


# In[8]:


Y_train = Y_to_softmax(Y_train)
Y_test = Y_to_softmax(Y_test)
print(Y_train.shape)
print(Y_test.shape)


# In[9]:


#print(np.max(X_train))
#print(np.max(X_test))
X_train = X_train/np.max(X_train)
X_test = X_train/np.max(X_test)


# In[10]:


def result_cleaning(result):
    result = result == np.max(result,axis = 0)
    result = result*1
    return result


# In[11]:


def accuracy_calculator(result,Y):
    n = Y.shape[0]*Y.shape[1]
    accuracy = np.sum(result == Y)/n
    return accuracy


# In[12]:


""" Activation functions and their derivatives w.r.t input"""

def relu(z):
    return np.maximum(0,z)

def relu_backward(z):
    dz = (z>0)*1
    return dz

def softmax(z):
    a = np.power(math.e,z)
    a /= np.sum(a,axis = 0)
    return a

def softmax_backward(z):
    a = softmax(z)
    dz = a - np.power(a,2)
    return dz
def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a
def sigmoid_backward(z):
    a = sigmoid(z)
    dz = a*(1-a)
    return dz


# In[13]:


z = np.array([-1,2,3,-3,2,-1])
z = np.reshape(z,(2,3))
print(relu(z))


# In[14]:


def initialize_parameters(layer_dims):
    """ layer_dims = list containing no of hidden units of each layer from input to output"""
    parameters = {}
    for l in range(len(layer_dims)-1):
        parameters["W" + str(l+1)] = np.random.randn(layer_dims[l+1],layer_dims[l])*np.sqrt(2/layer_dims[l])
        parameters["b" + str(l+1)] = np.zeros((layer_dims[l+1],1))
        
        assert(parameters["W" + str(l+1)].shape == (layer_dims[l+1], layer_dims[l]))
        assert(parameters["b" + str(l+1)].shape == (layer_dims[l+1], 1))
    
    return parameters


# In[15]:


para = initialize_parameters([3,4,2])
print(para)
print(np.var(para["W1"]))


# In[16]:


def linear_forward(A,W,b):
    """
    A - Matrix of activations from previous layer of N.N of shape (layer_dims[l-1], no of examples)
    W - Matrix of weights corresponding to current layer of shape (layer_dims[l],layer_dims[l-1])
    b - Matrix of biases corresponding to current layer of shape (layer_dims[l],1)
    """
    
    Z = np.dot(W,A) + b
    return Z


# In[17]:


def activation_forward(A_prev,W,b,activation_function):
    """
    A_prev - Matrix of activations from previous layer of N.N of shape (layer_dims[l-1], no of examples)
    W - Matrix of weights corresponding to current layer of shape (layer_dims[l],layer_dims[l-1])
    b - Matrix of biases corresponding to current layer of shape (layer_dims[l],1)
    activation_function - Choice of our activation functions
    """
    Z = linear_forward(A_prev,W,b)
    if activation_function == "relu":
        A = relu(Z)
    elif activation_function == "softmax":
        A = softmax(Z)
    elif activation_function == "sigmoid":
        A = sigmoid(Z)
        
    return A,Z   


# In[18]:


def forward_propagation(X,parameters):
    """
    X - Matrix of inputs of shape (no. of features, no. of examples)
    parameters - Dictionary containing parameters for every layer 
    """
    L = len(parameters)//2
    activations = {}
    linears = {}
    activations["A0"] = X
    for l in range(1,L):
        activations["A" + str(l)],linears["Z" + str(l)] = activation_forward(activations["A" + str(l-1)],parameters["W"+str(l)],parameters["b"+str(l)],"relu")
    activations["A" + str(L)],linears["Z" + str(L)] = activation_forward(activations["A" + str(L-1)],parameters["W" + str(L)],parameters["b"+str(L)],"softmax")
    #activations["A" + str(L)][activations["A" + str(L)] == float(0.0)] += np.power(10.0,-10)
    #activations["A" + str(L)][activations["A" + str(L)] == float(1.0)] -= np.power(10.0,-10)
    return activations,linears


# In[19]:


x = np.array([1,-4,5,3,-2,-4,5,6,7])
x = np.reshape(x,(3,3))
para = initialize_parameters([3,4,2])
ac,lin = forward_propagation(x,para)
print(ac)
print(lin)
print(np.sum(ac["A2"],axis =0))


# In[20]:


def Frobenius_norm_square(parameters):
    L = len(parameters)//2
    s = 0
    for l in range(L):
        s = s + np.power(np.linalg.norm(parameters["W" + str(l+1)]),2)
    return s


# In[21]:


def compute_cost(Y_hat,Y,parameters,lambd):
    """
    Y_hat = Matrix of output of N.N,same as matrix activations["A"+str(L)] of shape (no.of classes, no of examples)
    Y = Matrix of true values of training examples of shape (no. of classes, no of examples)
    """
    m = Y.shape[1]
    #Y_hat[Y_hat == float(0.0)] += np.power(10.0,-10)
    #Y_hat[Y_hat == float(1.0)] -= np.power(10.0,-10)
    prop = np.log(Y_hat)
    pron = np.log(1-Y_hat)
    pos = np.sum(np.multiply(prop,Y))
    neg = np.sum(np.multiply(pron,1-Y))
    cost = (-1/m)*(pos+neg) + (lambd/(2*m))*Frobenius_norm_square(parameters)
    return cost
        


# In[22]:


Y = np.array([0,1,0,1,0,0])
Y = np.reshape(Y,(2,3))
Y_hat = np.array([1.0, 0.0, 0.0,1.0, 0., 0.])
Y_hat = np.reshape(Y_hat,(2,3))
cost = compute_cost(Y_hat,Y,para,0.1)
print(cost)
print(np.divide(1-Y,1-Y_hat))
#yn = 1-Y_hat
#print(1-Y_hat)
#print(np.multiply(prop,Y))
#print(pron)
#print(neg)
#pos =  np.sum(np.multiply(np.log(Y_hat),Y))
#print(pos)
#print(np.log(Y_hat))
#print(np.log(1 - Y_hat))


# In[23]:


def linear_backward(dZ, A, W,lambd):
    """
    dZ - Gradient of linear functions of layer l w.r.t cost of shape same as Z of layer l (layer_dims[l], no of examples)
    A - Activations of previous layer i.e activations["A" + str(l-1)] of shape (layer_dims[l-1], no of examples)
    W - Matrix of weights of layer l of shape (layer_dims[l], layer_dims[l-1])
    """
    """
    Returns
    dW - Gradient of weights of current layer w.r.t cost and same shape as W.
    dA - Gradients of activations of previous layer w.r.t cost and same shape as A.
    db - Gradients of biases of current layer w.r.t cost and same shape as parameters["b" + str(l)]
    """
    m = dZ.shape[1]
    dW = (1/m)*np.dot(dZ,A.T) + (lambd/m)*W
    db = (1/m)*np.sum(dZ, axis =1, keepdims = True)
    dA = np.dot(W.T, dZ)
    
    return dW,dA,db


# In[24]:


def activation_backward(dA,Z,A_prev,W,activation_function,lambd):
    """
    dA - Matrix of gradients of activations of current layer l w.r.t cost and of shape (layer_dims[l],no.of examples)
    activation_function = Activation function used in layer l during forward propagation.
    Z - Matrix of linears of current layer l and of shape (layer_dims[l], no of examples)
    A_prev - Matrix of activation of layer l-1 of shape (layer_dims[l-1], no of examples)
    W - Matrix of weights of layer l of shape (layer_dims[l], layer_dims[l-1])
    """
    """
    Returns
    dW - Gradient of weights of current layer w.r.t cost and same shape as W.
    dA_prev - Gradients of activations of previous layer w.r.t cost and same shape as A_prev.
    db - Gradients of biases of current layer w.r.t cost and same shape as parameters["b" + str(l)]
    """
    
    if activation_function == "relu":
        dZ = np.multiply(dA,relu_backward(Z))
        dW,dA_prev,db = linear_backward(dZ, A_prev, W,lambd)
    elif activation_function == "softmax":
        dZ = np.multiply(dA, softmax_backward(Z))
        dW,dA_prev,db = linear_backward(dZ,A_prev,W,lambd)
    elif activation_function == "sigmoid":
        dZ = np.multiply(dA, sigmoid_backward(Z))
        dW,dA_prev,db = linear_backward(dZ,A_prev,W,lambd)
        
    return dW, dA_prev, db


# In[25]:


def backward_propagation(Y,activations,parameters,linears,lambd):
    """
    dAL - Gradient of activations of layer L w.r.t cost
    Y - Matrix of true values of the training set
    activations - Dictionary containing the values of activations of every layer
    parameters - Dictionary containing values of parameters for every layer
    """
    grads = {}
    L = len(parameters)//2
    AL = activations["A" + str(L)]
    m = AL.shape[1]
    #AL[AL == 0.0] += np.power(10.0,-10)
    #AL[AL == 1.0] -= np.power(10.0,-10)
    grads["dA"+str(L)] = -(np.divide(Y,AL) - np.divide(1-Y,1-AL))
    grads["dW" + str(L)],grads["dA" + str(L-1)],grads["db"+ str(L)] = activation_backward(grads["dA"+str(L)],linears["Z"+str(L)],activations["A"+str(L-1)],parameters["W"+str(L)],"sigmoid",lambd) 
    
    for l in reversed(range(1,L)):
        grads["dW" + str(l)],grads["dA" + str(l-1)],grads["db"+ str(l)] = activation_backward(grads["dA"+str(l)],linears["Z"+str(l)],activations["A"+str(l-1)],parameters["W"+str(l)],"relu",lambd) 
   
    return grads
        


# In[26]:


def update_parameters(parameters,grads,learning_rate):
    """
    parameters - Dictionary containing parameters of every layer
    grads - Dictionary containing gradients of parameters w.r.t cost for every layer
    """
    L = len(parameters)//2
    for l in range(L):
        parameters["W"+ str(l+1)] = parameters["W"+ str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b"+ str(l+1)] = parameters["b"+ str(l+1)] - learning_rate*grads["db" + str(l+1)]
    return parameters


# In[27]:


def mini_batch_divider(X,Y,mbs):
    """
    X - Matrix of inputs to neural network of shape (no of features, no of examples)
    Y - Matrix of true values of ouptuts of neural network of shape (no of classes, no of examples)
    mbs - Size of minibatch
    """
    m = Y.shape[1]
    np.random.shuffle(X.T)
    np.random.shuffle(Y.T)
    minibatches = []
    n = math.floor(m/mbs)
    for i in range(n):
        X_mini = X[:, i*mbs:(i+1)*mbs]
        Y_mini = Y[:, i*mbs:(i+1)*mbs]
        minibatches.append((X_mini,Y_mini))
    if m%mbs != 0:
        X_mini_last = X[:, n*mbs:]
        Y_mini_last = Y[:, n*mbs:]
        minibatches.append((X_mini_last,Y_mini_last))
    
    return minibatches


# In[28]:


def Neural_network_complete_model(X_train,Y_train,layer_dims,learning_rate,num_iterations,lambd,mbs):
    """
    X_train - Matrix of inputs to neural network and of shape (no. of features,no of examples)
    Y_train - Matrix of true values of Neural network outputs and of shape (no of classes, no of examples)
    layer_dims - List of no of units in each layer of neural network
    lambd - Value of L2 regularization cost constant
    mbs - mini_batch size to be used in training
    """
    
    parameters = initialize_parameters(layer_dims)
    cost_list = []
    #print(parameters)
    L = len(layer_dims)-1
    minibatches = mini_batch_divider(X_train,Y_train,mbs)
    
    for i in range(num_iterations):
        #learning_rate = np.power(0.95,i)*learning_rate
        for minibatch in minibatches:
            X_mini = minibatch[0]
            Y_mini = minibatch[1]
            activations,linears = forward_propagation(X_mini,parameters)
            cost = compute_cost(activations["A"+str(L)], Y_mini,parameters,lambd)
            grads = backward_propagation(Y_mini,activations,parameters,linears,lambd)
            parameters = update_parameters(parameters,grads,learning_rate)
        if i%200 == 0:
            cost_list.append(cost)
            print("Cost after iteration %i : %f, lr = %f" %(i,cost,learning_rate)) 
        
            
    plt.plot(cost_list)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning Rate = " + str(learning_rate) + " M.B.S = " + str(mbs))
    plt.show()
    return parameters
        


# In[ ]:


layer_dims = [784,26,26,10]
parameters = Neural_network_complete_model(X_train,Y_train,layer_dims,0.01,2500,0.0, Y_train.shape[1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




