# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 09:34:42 2019

@author: Jeyan
"""
#Import packages

import numpy as np
import struct
import numpy as np
import os
os.chdir(r'C:\\Users\\murug\\Desktop\\Jey\\GMU\\Semester 3\\OR 610\\mnist')


# Read the file using the given function

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


x_train =  read_idx('train-images.idx3-ubyte')
y_train  = read_idx('train-labels.idx1-ubyte')
x_test  = read_idx('t10k-images.idx3-ubyte')
y_test  = read_idx('t10k-labels.idx1-ubyte')

x_train.shape
y_train.shape
x_test.shape
y_test.shape

#Shapes of the test and train dataset

print('Train: x=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: x=%s, y=%s' % (x_test.shape, y_test.shape))

len(x_train)

# Reshaping the dataset into a compatible format since the size and shape 
# of the arrays are different and not compatible with each other 

num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape((x_train.shape[0], num_pixels)).astype('float32')
x_test = x_test.reshape((x_test.shape[0], num_pixels)).astype('float32')

x_train = x_train / 255
x_test = x_test / 255

# One hot encoding of the y train and y test dataset

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


# Sigmoid function to introduce the non linearity to the model 

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def sigmoid2(s):
    return s * (1 - s)

# In order to calculate the probability distribution of the range and 
# to normalize the data
    
def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss

# This class runs the feedforward and backpropagation fucntion
# The number of neurons is also mentioned for the sinlge hidden layer 
# as 128. The learning rate for different values is given as 
# 0.5,1,1.5. 

class Neuralnet:
    def __init__(self, x, y):
        self.x = x
        neurons = 128
        self.lr = 0.5
        #self.lr = 1
        #self.lr = 1.5
        ip_dim = x.shape[1]
        op_dim = y.shape[1]

        self.w1 = np.random.randn(ip_dim, neurons)
        self.b1 = np.zeros((1, neurons))
        self.w2 = np.random.randn(neurons, neurons)
        self.b2 = np.zeros((1, neurons))
        self.w3 = np.random.randn(neurons, op_dim)
        self.b3 = np.zeros((1, op_dim))
        self.y = y

    def feedforward(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)
        
    def backprop2(self):
        loss = error(self.a3, self.y)
        print('Error :', loss)
        a3_delta = cross_entropy(self.a3, self.y) # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid2(self.a2) # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid2(self.a1) # w1
        
        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta) 
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta) 
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)
        

    def predict(self, data):
        self.x = data
        self.feedforward()
        return self.a3.argmax()
    
    # This model runs the class by taking both of the input train functions. 
    
model = Neuralnet(x_train/16.0, np.array(y_train))

# For maximum iterations, we given 1500 epoch since the lesser number of 
# epoch would reduce the chances of error. 

epochs = 1000
for x in range(epochs):
    model.feedforward()
    model.backprop()

# This fucntion gives us the accuracy for the model . 
    
def get_acc(x, y):
    acc = 0
    for xx,yy in zip(x, y):
        s = model.predict(xx)
        if s == np.argmax(yy):
            acc +=1
    return acc/len(x)*100

# The testing and training accuracy of the model is tested. 
    
print("Training accuracy : ", get_acc(x_train/16, np.array(y_train)))
print("Test accuracy : ", get_acc(x_test/16, np.array(y_test)))

# The training speed for 1000 iterations is 75 mins
	



                                        #################

                                        #################

                                        #################

                                        #################

                                        #################

# Question : 2

# The following function has L2 regularization where the value of lambda is 0.1
# The value of training and testing accuracy is changed when we add regularization parameter. 

class Neuralnet2:
    def __init__(self, x, y):
        self.x = x
        neurons = 128
        self.lr = 0.5
        #self.lr = 1
        #self.lr = 1.5
        ip_dim = x.shape[1]
        op_dim = y.shape[1]

        self.w1 = np.random.randn(ip_dim, neurons)
        self.b1 = np.zeros((1, neurons))
        self.w2 = np.random.randn(neurons, neurons)
        self.b2 = np.zeros((1, neurons))
        self.w3 = np.random.randn(neurons, op_dim)
        self.b3 = np.zeros((1, op_dim))
        self.y = y

    def feedforward2(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)
        
    def backprop2(self):
        lambda_v = 0.1
        samples = len(x_train)
        s1 = np.sum(self.w1*self.w1)
        s2 = np.sum(self.w2*self.w2)
        reg = (lambda_v/(2*samples))*(s1 + s2)
        loss = error(self.a3, self.y) + reg
        print('Error :', loss)
        a3_delta = cross_entropy(self.a3, self.y) # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid2(self.a2) # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid2(self.a1) # w1

        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

    def predict(self, data):
        self.x = data
        self.feedforward2()
        return self.a3.argmax()
    

# This model runs the class by taking both of the input train functions. 
    
model_2 = Neuralnet2(x_train/16.0, np.array(y_train))

# For maximum iterations, we given 1000 epoch since the lesser number of 
# epoch would reduce the chances of error. 

epochs = 1000
for x in range(epochs):
    model_2.feedforward2()
    model_2.backprop2()
    
    
def get_acc_1(x, y):
    acc = 0
    for xx,yy in zip(x, y):
        s = model_2.predict(xx)
        if s == np.argmax(yy):
            acc +=1
    return acc/len(x)*100


# The testing and training accuracy of the model is tested. 
    
print("Training accuracy : ", get_acc_1(x_train/16, np.array(y_train)))
print("Test accuracy : ", get_acc_1(x_test/16, np.array(y_test)))

# The training speed for 1000 iterations is 70 mins

# On comparision, we could see the training speed when regularization is 
# added is lower than when not. This proves that the higher accuracy
# is achieved when l2 reg. term is added since the loss function is 
# reduced. 

