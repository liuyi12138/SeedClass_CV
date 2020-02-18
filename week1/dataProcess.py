import numpy as np
import matplotlib.pyplot as plt

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
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float") # reshape to (10000, 32, 32, 3)
    Y = np.array(Y)
    return X, Y

def loadAll(path):
    prefix = 'data_batch_'
    valid_idx = 5
    for surfix in range(1, 7):
        if surfix == 1:
            x_train, y_train = loadOne(path + '/' + prefix + str(surfix))
        elif surfix == valid_idx:
            x_valid, y_valid = loadOne(path + '/' + prefix + str(surfix))
        else:
            X, Y = loadOne(path + '/' + prefix + str(surfix))
            x_all = np.concatenate((x_all, X), axis = 0)
            y_all = np.concatenate((y_all, Y), axis = 0)
    del X, Y
    x_test, y_test = loadOne(path + '/test_batch')
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def loadData(filename):
    dataDict = unpickle(filename)
    data = dataDict[ b'data']
    labels = np.array(dataDict[b'labels'])
    return data, labels

def plotSample():
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    sampleNum = 10
    plt.figure(figsize=(12,12))
    figureData = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)  #对图像数据重新分割
    for label,classname in enumerate(classes):    #label返回元素位置，classname返回分类名称
        indexArray = np.where(y == label)[0]
        for i in range(sampleNum):    #选取前sampleNum张图片
            pltIndex = i * len(classes) + label + 1  #计算图片位置
            plt.subplot(sampleNum,len(classes),pltIndex) #选择图片位置
            plt.imshow(figureData[indexArray[i]])    #绘制图像
            plt.axis('off')  #隐藏坐标轴
            if i == 0:
                plt.title(classname)

if __name__ == '__main__':
    x, y = loadData("D:\HUST\寒假课程资料\数字图像处理\课设\week1\cifar-10-batches-py\data_batch_1")
    print(x.shape)
    print(y.shape)
    plotSample()

