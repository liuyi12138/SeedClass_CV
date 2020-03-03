#!/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.decomposition import PCA
from skimage.feature import hog, local_binary_pattern
from skimage import color, filters

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
try:
    from config import batchBasePath, projectPath, datasetDir
except:
    print("[ERROR] Please create your own config.py out of configTemplate.py before proceeding!")
    exit(0)


def unpickle(filename):
    """
    data_dict: a object consists of data and labels
    """
    import pickle
    with open(filename, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def loadOne(filename):
    # load single batch of cifar dataset
    datadict = unpickle(filename)
    data = datadict[b'data']
    labels = np.array(datadict[b'labels'])
    return data, labels


def loadAll(valid_idx=None):
    """
    valid_idx: choose one batch of those as validation set
    """
    if valid_idx == None:
        valid_idx = 0

    data_train, labels_train = [], []
    for suffix in range(1, 6):
        if suffix == valid_idx:
            data_valid, labels_valid = loadOne(batchBasePath + str(suffix))
        else:
            data_batch, labels_batch = loadOne(batchBasePath + str(suffix))
            data_train = np.concatenate((data_train, data_batch), axis=0)
            labels_train = np.concatenate((labels_train, labels_batch), axis=0)

    data_test, labels_test = loadOne(datasetDir + '/test_batch')
    return data_train, labels_train, data_valid, labels_valid, data_test, labels_test


def plotSample(data, labels):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_samples = 10
    plt.figure(figsize=(12, 12))
    figure_data = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)  # 对图像数据重新分割
    for label, classname in enumerate(classes):  # label返回元素位置，classname返回分类名称
        index_array = np.where(labels == label)[0]
        for i in range(num_samples):  # 选取前sampleNum张图片
            plt_index = i * len(classes) + label + 1  # 计算图片位置
            plt.subplot(num_samples, len(classes), plt_index)  # 选择图片位置
            plt.imshow(figure_data[index_array[i]])  # 绘制图像
            plt.axis('off')  # 隐藏坐标轴
            if i == 0:
                plt.title(classname)
    plt.show()


def pca(data_train, data_test, n_components):
    """
    x_train:
    """
    pca_data_prefix = "/pca_data/"
    pca_dir = os.path.dirname(os.path.realpath(__file__)) + pca_data_prefix
    if os.path.exists(pca_dir + str(n_components) + '.npy') == True:
        data_train_pca = np.load(pca_dir + str(n_components) + '.npy')
        data_test_pca = np.load(pca_dir + 'test.npy')
    else:
        pca = PCA(n_components=n_components)
        pca.fit(data_train)
        data_train_pca = pca.transform(data_train)
        data_test_pca = pca.transform(data_test)
        np.save((pca_dir + str(n_components) + '.npy'), data_train_pca)
        np.save(pca_dir + 'test.npy', data_test_pca)
    return data_train_pca, data_test_pca


# Hog处理 注意更换一下地址
def Hog(data_train, data_test, f):
    hog_data_prefix = "/hog/"
    hog_dir = os.path.dirname(os.path.realpath(__file__)) + hog_data_prefix
    if os.path.exists(hog_dir + str(f) + '.npy') == True:
        data_train_hog = np.load(hog_dir + str(f) + '.npy')
        data_test_hog = np.load(hog_dir + 'test.npy')
    else:
        figure_data_train = data_train.reshape(len(data_train), 3, 32, 32).transpose(0, 2, 3, 1)
        data_train_hog = []
        for i in range(len(data_train)):
            data_train_hog.append(hog(figure_data_train[i], orientations=12, pixels_per_cell=(2, 2), cells_per_block=(1, 1),
                                   visualise=False))
        data_train_hog = np.array(data_train_hog)
        np.save(hog_dir + str(f) + '.npy', data_train_hog)

        figure_data_test = data_test.reshape(len(data_train), 3, 32, 32).transpose(0, 2, 3, 1)
        data_test_hog = []
        for i in range(len(data_test)):
            data_test_hog.append(hog(figure_data_test[i], orientations=12, pixels_per_cell=(2, 2), cells_per_block=(1, 1),
                                  visualise=False))
        data_test_hog = np.array(data_test_hog)
        np.save(hog_dir + 'test.npy', data_test_hog)
    return data_train_hog, data_test_hog


# 转灰度图
def ToGray(data):
    data_gray = []
    figure_data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)
    for i in range(len(figure_data)):
        img_gray = color.rgb2gray(figure_data[i])
        image = filters.roberts(img_gray)
        image = image.reshape(1024)
        data_gray.append(image)
    data_gray = np.array(data_gray)
    return data_gray


# LBP局部特征提取
def LBP(data):
    figure_data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)
    data_lbp = []
    for i in range(len(data)):
        img_gray = color.rgb2gray(figure_data[i])
        image_lbp = local_binary_pattern(img_gray, 8, 1)
        image_lbp = image_lbp.reshape(1024)
        data_lbp.append(image_lbp)
    data_lbp = np.array(data_lbp)
    return data_lbp


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


# 绘制曲线图
def plotK(data_k):
    data = []
    for i in range(len(data_k)):
        data.append(float(format(data_k[i], '.3f')))
    x = range(1, len(data) + 1)
    max_indx = np.argmax(data)
    show_max = '[' + str(max_indx + 1) + ' ' + str(data[max_indx]) + ']'
    plt.plot(x, data_k, color='red', label='Hog&L1')
    plt.legend()  # 显示图例
    plt.xlabel('K')
    plt.ylabel('accurity')
    plt.annotate(show_max, xytext=(max_indx, data[max_indx]), xy=(max_indx, data[max_indx]))
    plt.show()


def getValid():
    data, labels = loadOne(batchBasePath + "5")
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    sampleNum = 100
    x_test = []
    y_test = []
    for label, classname in enumerate(classes):  # label返回元素位置，classname返回分类名称
        x_class = []
        y_class = []
        indexArray = np.where(labels == label)[0]
        for i in range(sampleNum):  # 选取前sampleNum张图片
            x_class.append(data[indexArray[i]])
            y_class.append(labels[indexArray[i]])
        x_test.append(x_class)
        y_test.append(y_class)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    valid_path = projectPath + '/valid/'
    np.save(valid_path + 'x.npy', x_test)
    np.save(valid_path + 'y.npy', y_test)


if __name__ == "__main__":
    x, y = loadOne(batchBasePath + "1")
    print(x.shape)
    print(y.shape)
    plotSample(x, y)
