#!/bin/env python3
import numpy as np
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def loadOne(filename):
    # load single batch of cifar
    datadict = unpickle(filename)
    X = datadict[b'data']
    Y = datadict[b'labels']
    Y = np.array(Y)
    return X, Y

def loadAll(path, valid_idx = None):
    prefix = 'data_batch_'
    if valid_idx == None:
        valid_idx = 0
    cnt = 0
    x_valid = np.array([])
    y_valid = np.array([])
    for surfix in range(1, 6):
        if surfix == valid_idx:
            x_valid, y_valid = loadOne(path + '/' + prefix + str(surfix))
        elif cnt == 0:
            x_train, y_train = loadOne(path + '/' + prefix + str(surfix))
            cnt = 1
        else:
            X, Y = loadOne(path + '/' + prefix + str(surfix))
            x_train = np.concatenate((x_train, X), axis = 0)
            y_train = np.concatenate((y_train, Y), axis = 0)
    del X, Y
    x_test, y_test = loadOne(path + '/test_batch')
    return x_train, y_train, x_valid, y_valid, x_test, y_test

