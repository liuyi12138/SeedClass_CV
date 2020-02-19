#!/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

if not os.path.exists("config.py"):   
    print("[ERROR] Please create your own config.py out of configTemplate.py before proceeding!")
    exit(0)
from config import dataDir, testDataDir, trainDataDir

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

def loadData(filename):
    dataDict = unpickle(filename)
    data = dataDict[ b'data']
    labels = np.array(dataDict[b'labels'])
    return data, labels

def sample(data):
    length = data.shape[0]
    data = data.reshape(length, 3, 32, 32)
    dataSample = data[:,:,::2,::2]
    dataSample = np.reshape(dataSample,(dataSample.shape[0],-1))
    return dataSample

def pca(x_train, x_test):
    pca=PCA(n_components=100)
    pca.fit(x_train)
    x_train_new = pca.transform(x_train)
    x_test_new = pca.transform(x_test)  
    return x_train_new,x_test_new

def plotSample(data, labels):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    sampleNum = 10
    plt.figure(figsize=(12,12))
    figureData = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)  #对图像数据重新分割
    for label,classname in enumerate(classes):    #label返回元素位置，classname返回分类名称
        indexArray = np.where(labels == label)[0]
        for i in range(sampleNum):    #选取前sampleNum张图片
            pltIndex = i * len(classes) + label + 1  #计算图片位置
            plt.subplot(sampleNum,len(classes),pltIndex) #选择图片位置
            plt.imshow(figureData[indexArray[i]])    #绘制图像
            plt.axis('off')  #隐藏坐标轴
            if i == 0:
                plt.title(classname)

if __name__ == "__main__":
    x, y = loadData(dataDir)
    print(x.shape)
    print(y.shape)
    plotSample(x,y)
