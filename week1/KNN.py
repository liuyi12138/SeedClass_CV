import os
import numpy as np
from dataProcess import loadAll, pca
from configTemplate import dataDir
from math import pow
from scipy.spatial.distance import cosine


def LmNormMetric(norm_argument):
    def LmNorm(x1, x2):
        dis = []
        for i in range(x1.shape[0]):
            abs_list = np.abs(x1[i] - x2)
            msum = np.sum(abs_list ** norm_argument)  # pow m and sum
            dis.append(pow(msum, 1 / norm_argument))  # pow 1/m
        return np.array(dis)
    return LmNorm

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

def cosDis(x1, x2):
    # x1 should be the xtr matrix, ndarray
    # x2 should be a row of the xpred matrix, list
    # return: dis should also be a ndarray
    dis = []
    for i in range(x1.shape[0]):
        dis.append(cosine(x1[i], x2))
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


def getDistances(x1, x2, metric, name_tag):
    # Parameters Explanation:
    #   @x1, x2:        numpy 2D-matrixes, x1 is train, x2 is valid
    #   @weights:       A list indicating the weights distribution of cosine, L1 and L2 distance
    #   @data_type:     1-5, 1 for raw, 2 for Sample Grey, 3 for PCA, 4 for HOG, 5 for GreyHOG
    total_path = 'distances_' + str(name_tag) + '.npy'  # total_dis = weights_matrix x distances_matrix

    distances_matrix = []
    num_test = x2.shape[0]

    # get cos, L1 and L2
    if os.path.exists(total_path):
        print('%s detected, load npy data' % total_path)
        distances_matrix = np.load(total_path)
    else:
        print('Start to create npy!')
        for i in range(num_test):
            if (i + 1) % 100 == 0:
                print('%d of %d finished' % ((i + 1), num_test))
            distances_matrix.append(metric(x1, x2[i]))
        np.save(total_path, np.array(distances_matrix))
        print('L2 finished')

    return np.array(distances_matrix)


class KNearestNeighbor:
    def __init__(self):
        return None         # don't leave it "pass"

    def train(self, x, y):
        self.xtr = x
        self.ytr = y

    def predict(self, x, k = 10, valid_idx = 5, metric=LmNormMetric(1)):
        self.value_k = k
        self.valid_idx = valid_idx

        print('\nStart to process\n')
        num_test = x.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        self.dis_weights = [1, 0, 0]
        distances_matrix = getDistances(self.xtr, x, metric=metric, name_tag=k)
        for i in range(num_test):
            indexs = np.argsort(distances_matrix[i])    #对index排序
            closestK = self.ytr[indexs[:k]]             #取距离最小的K个点的标签值
            count = np.bincount(closestK)               #获取各类的得票数
            Ypred[i] = np.argmax(count)                 #找出得票数最多的一个
            # if (i+1) % 10 == 0:
            #    print('now: %d/%d, Ypred[%d] = %d\r' % (i+1, num_test, i, Ypred[i]))
        return Ypred

    #此处lm整理一下？
    def predict_2Class(self, test_data, k, m, f):
        num_test = test_data.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        distances = self.getDistance(m,test_data,f)
        
        for i in range(num_test):
            indexs = np.argsort(distances[i]) #对index排序
            closestK = self.ytr[indexs[:k]] #取距离最小的K个点
            count = np.bincount(closestK) #获取各类的得票数
            Ypred[i] = np.argmax(count) #找出得票数最多的一个
        return Ypred
    
    def predict_5Class(self,test_data, k, m, f, result_2Class):
        num_test = test_data.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        distances = self.getDistance(m,test_data,f)
        
        for i in range(num_test):
            indexs = np.argsort(distances[i]) #对index排序
            allDis = self.ytr[indexs] #获取到所有数据
            deleteINdex = []
            if result_2Class[i] == 1:
                for j in range(1000):
                    if(allDis[j] not in [0,1,8,9]):
                        deleteINdex.append(j)
            else:
                for j in range(1000):
                    if(allDis[j]in [0,1,8,9]):
                        deleteINdex.append(j)
            allDis = np.delete(allDis,deleteINdex)
            closestK = allDis[:k] #取前K个点
            count = np.bincount(closestK) #获取各类的得票数
            Ypred[i] = np.argmax(count) #找出得票数最多的一个
        return Ypred
    
    def evaluate(self, Ypred, y):
        num_test = len(y)
        num_correct = np.sum(Ypred == y)
        accuracy = float(num_correct) / num_test
        print("[cos, L1, L2] = ", self.dis_weights, "With k = %d, %d / %d correct => accuracy: %.2f %%" %(self.value_k, num_correct, num_test, accuracy*100)) 
        return accuracy

if __name__ == "__main__":
    valid_idx = 5
    x_train, y_train, x_valid, y_valid, x_test, y_test = loadAll(valid_idx)
    
    x_valid = np.load(dataDir + '/x.npy').reshape(1000, 3072)
    y_valid = np.load(dataDir + '/y.npy').reshape(1000,)


    xtr_new, xva_new = pca(x_train, x_valid, n_components = 30)
    print(xva_new.shape)

    classifier = KNearestNeighbor()
    classifier.train(xtr_new, y_train)
    for k in range(1,101):
        result = classifier.predict(x = xva_new[:1000], k = k, valid_idx = valid_idx)
        classifier.evaluate(result, y_valid[:1000])
