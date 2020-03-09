#!/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from skimage import io,transform

#对list各元素取对数后标准化
def normalization(data):
    data = np.exp(data)
    _range = np.sum(data)
    return (data) / _range

def unpickle(filename):
    """
    data_dict: a object consists of data and labels
    """
    import pickle
    with open(filename, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def load_one(filename):
    # load single batch of cifar dataset
    datadict = unpickle(filename)
    data = datadict[b'data']
    labels = np.array(datadict[b'labels'])
    return data, labels

#图像归一化
def normalize_image(data):
    data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    data = data.reshape(data.shape[0], 3072)
    return np.array(data)/255