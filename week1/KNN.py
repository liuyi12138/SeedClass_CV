import numpy as np
from dataProcess import loadAll
from configTemplate import dataDir
from math import pow

def LmNorm(x1,x2,m):
    # x1 should be the xtr matrix, ndarray
    # x2 should be a row of the xpred matrix, list
    # return: dis should also be a ndarray
    dis = []
    for i in range(x1.shape[0]):
        abs_list = np.abs(x1[i] - x2)
        msum = np.sum(abs_list**m)          # pow m and sum
        # print(pow(psum, 1/m))
        dis.append(pow(msum, 1/m))          # pow 1/m
    #print(dis)
    return np.array(dis)

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
            if (i+1) % 10 == 0:
                print('now: %d/%d' % (i+1, num_test))
        return Ypred

class KNearestNeighbor:
    def __init__(self):
        pass
    
    def train(self, x, y):
        self.xtr = x
        self.ytr = y
        
    def predict(self, x, k = None, m = None):
        if k == None:
            k = 10
        if m == None:
            m = 1
        self.value_k = k
        self.value_m = m
        #print('Start to predict')
        num_test = x.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        for i in range(num_test):
            distances = LmNorm(self.xtr, x[i], m)
            #distances = np.sum(np.abs(self.xtr - x[i]), axis=1)
            indexs = np.argsort(distances) #对index排序
            closestK = self.ytr[indexs[:k]] #取距离最小的K个点
            count = np.bincount(closestK) #获取各类的得票数
            Ypred[i] = np.argmax(count) #找出得票数最多的一个
            #if (i+1) % 100 == 0:
            #    print('now: %d/%d, Ypred[%d] = %d\r' % (i+1, num_test, i, Ypred[i]))
        return Ypred
    
    def evaluate(self, Ypred, y):
        num_test = len(y)
        num_correct = np.sum(Ypred == y)
        accuracy = float(num_correct) / num_test
        print('With k = %d, m = %d, %d / %d correct => accuracy: %.3f' %(self.value_k, self.value_m, num_correct, num_test, accuracy*100)) 
        return accuracy

if __name__ == "__main__":
    x_train, y_train, x_valid, y_valid, x_test, y_test = loadAll(dataDir, 1)
    #1w数据集 1k测试集
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    #5-NN
    classifier = KNearestNeighbor()
    classifier.train(x_train,y_train)
    result = classifier.predict(x_test,5,2)
    classifier.evaluate(result, y_test)
