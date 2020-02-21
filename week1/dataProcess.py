#!/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
from sklearn.decomposition import PCA
from skimage.feature import hog

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
try:
    from config import dataDir, batchBasePath, projectPath
except:
    print("[ERROR] Please create your own config.py out of configTemplate.py before proceeding!")
    exit(0)

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

#采样函数
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
    plt.show()

def pca(x_train, x_test, f):
    """
    x_train:
    """
    pca_path = projectPath + "/pca/"
    if os.path.exists(pca_path + str(f) + '.npy') == True:
        x_train_new = np.load(pca_path + str(f) + '.npy')
        x_test_new = np.load(pca_path + 'test.npy')
    else:
        pca = PCA(n_components=100)
        pca.fit(x_train)
        x_train_new = pca.transform(x_train)
        x_test_new = pca.transform(x_test)  
        np.save((pca_path + str(f) + '.npy'), x_train_new)
        np.save(pca_path + 'test.npy', x_test_new)
    return x_train_new,x_test_new

#Hog处理 注意更换一下地址
def Hog(x_train, x_test, f):
    hog_path = projectPath + "/hog/"
    if os.path.exists(hog_path + str(f) + '.npy') == True:
        x_train_new = np.load(hog_path  + str(f) + '.npy')
        x_test_new = np.load(hog_path + 'test.npy')
    else:
        figureData_train = x_train.reshape(len(x_train), 3, 32, 32).transpose(0, 2, 3, 1)
        x_train_new = []
        for i in range(len(x_train)):
            x_train_new.append(hog(figureData_train[i], orientations=12, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualise=False))
        x_train_new = np.array(x_train_new)
        np.save(hog_path + str(f) + '.npy', x_train_new)
        
        figureData_test = x_test.reshape(len(x_train), 3, 32, 32).transpose(0, 2, 3, 1)
        x_test_new = []
        for i in range(len(x_test)):
            x_test_new.append(hog(figureData_test[i], orientations=12, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualise=False))
        x_test_new = np.array(x_test_new)
        np.save(hog_path + 'test.npy', x_test_new)
    return x_train_new,x_test_new

#绘制曲线图
def plotK(dataK):
    data = []
    for i in range(len(dataK)):
        data.append(float(format(dataK[i],'.3f')))
    x = range(1,len(data) + 1)
    max_indx = np.argmax(data)
    show_max='['+str(max_indx+1)+' '+str(data[max_indx])+']'
    plt.plot(x, dataK, color='red', label='Hog&L1')
    plt.legend() # 显示图例
    plt.xlabel('K')
    plt.ylabel('accurity')
    plt.annotate(show_max,xytext=(max_indx,data[max_indx]),xy=(max_indx,data[max_indx]))
    plt.show()


def getValid():
    x, y = loadData(batchBasePath + "5")
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    sampleNum = 100
    figureData = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)  #对图像数据重新分割
    x_test = []
    y_test = []
    for label,classname in enumerate(classes):    #label返回元素位置，classname返回分类名称
        x_class = []
        y_class = []
        indexArray = np.where(y == label)[0]
        for i in range(sampleNum):    #选取前sampleNum张图片
            x_class.append(x[indexArray[i]])
            y_class.append(y[indexArray[i]])
        x_test.append(x_class)
        y_test.append(y_class)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    valid_path = projectPath + '/valid/'
    np.save(valid_path + 'x.npy', x_test)
    np.save(valid_path + 'y.npy', y_test)

if __name__ == "__main__":
    x, y = loadData(dataDir)
    print(x.shape)
    print(y.shape)
    plotSample(x,y)
