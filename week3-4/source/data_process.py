#!/bin/env python3
import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
from skimage import io,transform
from sklearn.decomposition import PCA
from skimage.feature import hog


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
    data = np.array(datadict[b'data'])
    labels = np.array(datadict[b'labels'])
    return data, labels

#图像归一化
def normalize_image(data):
    data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    data = data.reshape(data.shape[0], 3072)
    return np.array(data)/255

#对list各元素取对数后标准化
def normalization_1(data):
    data = np.exp(data)
    _range = np.sum(data)
    return (data) / _range

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def normalizationImage(data):
    data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    data = data.reshape(data.shape[0], 3072)
    data = np.array(data)/255
    data = data - 0.5
    return data

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

def pca(data_train, data_test, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data_train)
    data_train_pca = pca.transform(data_train)
    data_test_pca = pca.transform(data_test)
    return data_train_pca, data_test_pca

def Hog(data_train, data_test):
    figure_data_train = data_train.reshape(len(data_train), 3, 32, 32).transpose(0, 2, 3, 1)
    data_train_hog = []
    for i in range(len(data_train)):
        data_train_hog.append(hog(figure_data_train[i], orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                               visualize=False))
    data_train_hog = np.array(data_train_hog)

    figure_data_test = data_test.reshape(len(data_test), 3, 32, 32).transpose(0, 2, 3, 1)
    data_test_hog = []
    for i in range(len(data_test)):
        data_test_hog.append(hog(figure_data_test[i], orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                              visualize=False))
    data_test_hog = np.array(data_test_hog)
    return data_train_hog, data_test_hog


def LeakyRelu(x):
    for i in range(len(x)):
        if(x[i] < 0):
            x[i] *= 0.01
    return x

def LeakyRelu_derivative(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0.01
        else:
            x[i] = 1
    return x

def Elu(x):
    alpha = 1
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = alpha * (np.exp(x[i]) - 1)
    return x

def Elu_derivative(x):
    alpha = 1
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = alpha * np.exp(x[i])
        else:
            x[i] = 1
    return x