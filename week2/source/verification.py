#!/bin/env python
import numpy as np

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
        
        grad[ix] = ((fxh - fx) / h)[ix]
        it.iternext()
        
    return grad  

def analytic_grad(f, x):
    return f(x)

#f(x) = x^2
def f(x):
    return np.array(list(map(lambda num:num*num, x)))

#analytic_f(x) = 2*x
def analytic_f(x):
    return np.array(list(map(lambda num:2*num, x)))

# 验证analytic_grad的正确性
def verification():
    x = np.array([1.,2.,3.,4.])
    print("analytic_grad:       ", analytic_grad(analytic_f,x))
    print("numerical_gradient:  ", numerical_gradient(f, x))

verification()