# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 18:58:32 2021

@author: Agustin
"""

import numpy as np

def sigmoid(z):
        return 1/(1+np.exp(-z))
    
def dsig(z):
    return sigmoid(z)*(1-sigmoid(z))

def test(x, w11_1, w12_1, b1_1, w21_1, w22_1, b2_1, w11_2, w12_2, b1_2):
    j=0
    
    for sample in range(x.shape[0]):
        z1_1 = w11_1 * x[sample,1] + w12_1 * x[sample,0] + b1_1
        a1_1 = sigmoid(z1_1)
            
        z2_1 = w21_1 * x[sample,1] + w22_1 * x[sample,0] + b2_1
        a2_1 = sigmoid(z2_1)
            
        z1_2 = w11_2 * a1_1 + w12_2 * a2_1 + b1_2
        a1_2 = sigmoid(z1_2)
        print(z1_2)
        j += y[sample]-z1_2
    return j/x.shape[0]

x=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([0,1,1,0])

epochs = 1000

lr = 0.1

eps = 1

# Params Neurona 1 Layer 1
w11_1 = (np.random.rand() * (2*eps) - eps)
w12_1 = (np.random.rand() * (2*eps) - eps)
b1_1 = (np.random.rand() * (2*eps) - eps)

# Params Neurona 2 Layer 1
w21_1 = (np.random.rand() * (2*eps) - eps)
w22_1 = (np.random.rand() * (2*eps) - eps)
b2_1 = (np.random.rand() * (2*eps) - eps)

# Params Neurona 1 Layer 2
w11_2 = (np.random.rand() * (2*eps) - eps)
w12_2 = (np.random.rand() * (2*eps) - eps)
b1_2 = (np.random.rand() * (2*eps) - eps)

for epoch in range(epochs):
    for sample in range(x.shape[0]):
        #*****Forward*****
        # Neurona 1 Layer 1
        z1_1 = w11_1 * x[sample,1] + w12_1 * x[sample,0] + b1_1
        a1_1 = sigmoid(z1_1)
        
        # Neurona 2 Layer 1
        z2_1 = w21_1 * x[sample,1] + w22_1 * x[sample,0] + b2_1
        a2_1 = sigmoid(z2_1)
        
        # Neurona 1 Layer 2
        z1_2 = w11_2 * a1_1 + w12_2 * a2_1 + b1_2
        #a1_2 = sigmoid(z1_2)
        y_hat = z1_2
        
        #*****Loss Function*****
        J = (y[sample]-y_hat)**2
        
        dJ_w11_1 = -2*(y[sample]-y_hat) * w11_2 * dsig(z1_1) * x[sample,1]
        dJ_w12_1 = -2*(y[sample]-y_hat) * w11_2 * dsig(z1_1) * x[sample,0]
        dJ_b1_1 = -2*(y[sample]-y_hat) * w11_2 * dsig(z1_1)
        
        dJ_w21_1 = -2*(y[sample]-y_hat) * w12_2 * dsig(z2_1) * x[sample,1]
        dJ_w22_1 = -2*(y[sample]-y_hat) * w12_2 * dsig(z2_1) * x[sample,0]
        dJ_b2_1 = -2*(y[sample]-y_hat) * w12_2 * dsig(z2_1)
        
        dJ_w11_2 = -2*(y[sample]-y_hat) * a1_1
        dJ_w12_2 = -2*(y[sample]-y_hat) * a2_1
        dJ_b1_2 = -2*(y[sample]-y_hat)
        
        #*****Backward*****

        # Params Neurona 1 Layer 1
        w11_1 = w11_1 - lr * dJ_w11_1
        w12_1 = w12_1 - lr * dJ_w12_1
        b1_1 = b1_1 - lr * dJ_b1_1

        # Params Neurona 2 Layer 1
        w21_1 = w21_1 - lr * dJ_w21_1
        w22_1 = w22_1 - lr * dJ_w22_1
        b2_1 = b2_1 - lr * dJ_b2_1

        # Params Neurona 1 Layer 2
        w11_2 = w11_2 - lr * dJ_w11_2
        w12_2 = w12_2 - lr * dJ_w12_2
        b1_2 = b1_2 - lr * dJ_b1_2

j = test(x, w11_1, w12_1, b1_1, w21_1, w22_1, b2_1, w11_2, w12_2, b1_2)
