import numpy as np
from dataProcess import loadAll
from configTemplate import dataDir

class NearestNeighbor:
    def __init__(self):
        pass
    
    def train(self,x,y):
        self.xtr = x
        self.ytr = y
        
    def predict(self,x):
        num_test = x.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        for i in range(num_test):
            distances = np.sum(np.abs(self.xtr - x[i]), axis=1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]
            if i % 50 == 0:
                print('now: %d/%d' % (i, num_test))
        return Ypred

class KNearestNeighbor:
    def __init__(self):
        pass
    
    def train(self,x,y):
        self.xtr = x
        self.ytr = y
        
    def predict(self,x,k):
        num_test = x.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        for i in range(num_test):
            distances = np.sum(np.abs(self.xtr - x[i]), axis=1)
            indexs = np.argsort(distances) #对index排序
            closestK = self.ytr[indexs[:k]] #取距离最小的K个点
            count = np.bincount(closestK) #获取各类的得票数
            Ypred[i] = np.argmax(count) #找出得票数最多的一个
#             if i % 50 == 0:
#                 print('now: %d/%d' % (i, num_test))
        return Ypred

if __name__ == "__main__":
    x_train, y_train, x_valid, y_valid, x_test, y_test = loadAll(dataDir)
    #1w数据集 1k测试集
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    #5-NN
    classifier = KNearestNeighbor()
    classifier.train(x_train,y_train)
    result = classifier.predict(x_test,5)
    num_test= len(y_test)
    num_correct = np.sum(result == y_test)
    accuracy = float(num_correct) / num_test
    print('%d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
