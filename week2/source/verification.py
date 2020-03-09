#!/bin/env python
import numpy as np
from data_process import normalization

def numerical_gradient(f, x):
    fx = f(x)
    grad = np.zeros(x.shape)
    h = 0.00001
    #迭代数组，跟踪索引，可读可写
    it = np.nditer(x, flags = ['multi_index'] ,op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h
        fxh = f(x)
        x[ix] = old_value
        
        grad[ix] = ((fxh - fx) / h)
        it.iternext()
        
    return grad  

def analytic_grad(f, x):
    return f(x)

def numerical_grad(w):
    x = np.array([4,3,5])
    y = np.array([1,0])
    p = normalization(np.dot(x,w))
    loss = 0 - np.log(np.dot(y,p))
    return loss

def analytic_grad(w):
    x = np.array([4,3,5])
    y = np.array([1,0])
    p = normalization(np.dot(x,w))
    return np.array(np.dot(np.mat(x).T, np.mat(p-y)))

# 验证analytic_grad的正确性
def verification():
    w = np.array([[0.11,0.2],[0.2,0.53],[0.3,0.14]])
    print("analytic_grad:       \n", analytic_grad(w))
    print("numerical_gradient:  \n", numerical_gradient(numerical_grad, w))

verification()