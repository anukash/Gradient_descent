# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:01:06 2023

@author: Anurag
"""

# gradient descent for linear regression
#yhat = wx +b
#loss = (y-yhatt)**2/ N

import numpy as np

#initialize parameters
x = np.random.randn(10, 1)
y = 4*x +np.random.rand()

#parameters
w = 0.0
b = 0.0

#hyperparameter
learning_rate = 0.01

# create gradient descent function
def descent(x, y, w, b, learning_rate):
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]
    for xi, yi in zip(x, y):
        dldw += -2*xi*(yi- (w*xi +b))   ## differentiate loss = (y-(wx +b))**2/ N with respect to w
        dldb += -2*(yi- (w*xi +b))      ## differentiate loss = (y-(wx +b))**2/ N with respect to b
    
    #update weight and bias
    
    w = w - learning_rate*(1/N)*dldw
    b = b - learning_rate*(1/N)*dldb
    
    return w, b


#Iteratively make updates
for epoch in range(600):
    w, b = descent(x,y,w,b, learning_rate)
    yhat = w*x + b   # new pred
    loss = np.divide(np.sum((y - yhat)**2, axis=0), x.shape[0])
    print(f'{epoch} loss is {loss}, parameter w: {w}, b: {b}')
