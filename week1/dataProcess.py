#!/bin/env python3
import numpy as np
import os
from sklearn.decomposition import PCA
# from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage.feature import hog

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

def pca(x_train, x_valid, n_components = None): 
    if n_components == None:
        n_components = 30

    dir_name = './PCA'
    xtr_pca_name = '/xtr_n_' + str(n_components) + '.npy'
    xva_pca_name = '/xva_n_' + str(n_components) + '.npy'

    # verify dir
    if not os.path.exists(dir_name):
        print("'./PCA' doesn't exist, create one")
        os.makedirs(dir_name)
    else:
        print("'./PCA' detected, go on")



    if os.path.exists(dir_name + xtr_pca_name):
        print('PCA data detected, loading...')
        x_train_new = np.load(dir_name + xtr_pca_name)
        x_valid_new = np.load(dir_name + xva_pca_name)
    else:
        print("PCA data doesn't exist, creating...")
        pca=PCA(n_components = n_components)
        pca.fit(x_train)
        x_train_new = pca.transform(x_train)
        x_valid_new = pca.transform(x_valid)  
        np.save(dir_name + xtr_pca_name, x_train_new)
        np.save(dir_name + xva_pca_name, x_valid_new)
    print('Finish PCA\n')
    return x_train_new,x_valid_new

def Hog(x_train, x_test, f):
    hog_path = "./HOG/"
    para = f.split('_')
    [ori, p1, c1] = [int(para[0]), int(para[1][0]), int(para[2][0])]

    if os.path.exists(hog_path + str(f) + '.npy') == True:
        x_train_new = np.load(hog_path  + str(f) + '.npy')
        x_valid_new = np.load(hog_path + str(f) + '_valid.npy')
    else:
        figureData_train = x_train.reshape(len(x_train), 3, 32, 32).transpose(0, 2, 3, 1)
        x_train_new = []
        for i in range(len(x_train)):
            x_train_new.append(hog(figureData_train[i], orientations=ori, pixels_per_cell=(p1, p1), cells_per_block=(c1, c1), transform_sqrt=True))
        x_train_new = np.array(x_train_new)
        np.save(hog_path + str(f) + '.npy', x_train_new)
        
        figureData_test = x_test.reshape(len(x_test), 3, 32, 32).transpose(0, 2, 3, 1)
        x_valid_new = []
        for i in range(len(x_test)):
            x_valid_new.append(hog(figureData_test[i], orientations=ori, pixels_per_cell=(p1, p1), cells_per_block=(c1, c1), transform_sqrt=True))
        x_valid_new = np.array(x_valid_new)
        np.save(hog_path + str(f) + '_valid.npy', x_valid_new)
    return x_train_new, x_valid_new


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