import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from scipy.spatial.distance import cosine
from skimage.feature import hog
from skimage import color,filters
from PIL import Image
from skimage.feature import local_binary_pattern

def PCATest(x_train,x_test,y_train,y_test):
    acc_all = []
    for k in range(1,101):
        # x_train_hog, x_test_hog = Hog(x_train,x_test,1,False) #Hog
        x_train_new, x_test_new = pca(x_train,x_test,30) #pca
        classifier = KNearestNeighbor() #KNN
        classifier.train(x_train_new,y_train)
        result = classifier.predict_2Class(x_test_new,k,3,100)
            
        num_test= len(y_test)
        num_correct = np.sum(result == y_test) #计算准确率
        accuracy = float(num_correct) / num_test
        print("k:%d acc: %f" % (k, accuracy))
        acc_all.append(accuracy)


def HogTest(x_train,x_test,y_train,y_test):
    acc_all = []
    for k in range(1,101):
        x_train_hog, x_test_hog = Hog(x_train,x_test,1,False) #Hog
        # x_train_new, x_test_new = pca(x_train,x_test,30) #pca
        classifier = KNearestNeighbor() #KNN
        classifier.train(x_train_hog,y_train)
        result = classifier.predict_2Class(x_test_hog,k,3,100)
            
        num_test= len(y_test)
        num_correct = np.sum(result == y_test) #计算准确率
        accuracy = float(num_correct) / num_test
        print("k:%d acc: %f" % (k, accuracy))
        acc_all.append(accuracy)

def Two_FiveClassTest(x_train,x_test,y_train,y_test):
    y_2class_train = []
    for i in range(len(y_train)):
        if y_train[i] in [0,1,8,9]:
            y_2class_train.append(1)
        else:
            y_2class_train.append(0)
    y_2class_train = np.array(y_2class_train)

    y_2class_test = []
    for i in range(len(y_test)):
        if y_test[i] in [0,1,8,9]:
            y_2class_test.append(1)
        else:
            y_2class_test.append(0)
    y_2class_test = np.array(y_2class_test)

    x_train_hog, x_test_hog = Hog(x_train,x_test,1,False) #Hog
    classifier = KNearestNeighbor() #KNN
    classifier.train(x_train_hog,y_2class_train)
    result_2Class = classifier.predict_2Class(x_test_hog,15,1,100)

    acc_all = []
    for k in range(1,101):
        classifier.train(x_train_hog,y_train)
        result = classifier.predict_5Class(x_test_hog,k,1,100,result_2Class)
        
        num_test= len(y_test)
        num_correct = np.sum(result == y_test) #计算准确率
        accuracy = float(num_correct) / num_test
        print("k:%d acc: %f" % (k, accuracy))
        acc_all.append(accuracy)


def getData():
    for f in range(1,6):
        if f == 1:
            x_train, y_train = loadData("D:\HUST\寒假课程资料\数字图像处理\课设\week1\cifar-10-batches-py\data_batch_" + str(f))
        else:
            x_temp, y_temp = loadData("D:\HUST\寒假课程资料\数字图像处理\课设\week1\cifar-10-batches-py\data_batch_" + str(f))
            x_train = np.concatenate((x_train, x_temp), axis=0)
            y_train = np.concatenate((y_train, y_temp), axis=0)
    x_test, y_test = loadData("D:\HUST\寒假课程资料\数字图像处理\课设\week1\cifar-10-batches-py\\test_batch")
    return x_train,y_train,x_test,y_test