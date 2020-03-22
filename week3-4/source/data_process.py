#!/bin/env python3
import numpy as np
# import cupy as np
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

def loadOne(filename):
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

# 获取镜像数据
def getMirror(data, label):
    figure_data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)
    data_new = []
    label_new = []
    for i in range(len(data)):
        image = figure_data[i]
        image_mirror = image[:, ::-1]
        image = image.reshape(3 * 32 * 32)
        image_mirror = image_mirror.reshape(3 * 32 * 32)
        data_new.append(image)
        data_new.append(image_mirror)
        label_new.append(label[i])
        label_new.append(label[i])
    data_new = np.array(data_new)
    label_new = np.array(label_new)
    return data_new, label_new

def LeakyRelu(x):
    for i in range(len(x)):
        if(x[i] < 0):
            x[i] *= 0.01
    return x