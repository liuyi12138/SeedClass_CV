import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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

x, y = loadData("D:\HUST\寒假课程资料\数字图像处理\课设\week1\cifar-10-batches-py\data_batch_1")
print(x.shape)
print(y.shape)
plotSample()

