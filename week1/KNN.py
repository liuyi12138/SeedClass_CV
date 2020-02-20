import os
import numpy as np
from dataProcess import loadAll
from configTemplate import dataDir
from math import pow
from scipy.spatial.distance import cosine

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

def cosDis(x1,x2):
    # x1 should be the xtr matrix, ndarray
    # x2 should be a row of the xpred matrix, list
    # return: dis should also be a ndarray
    dis = []
    for i in range(x1.shape[0]):
        dis.append(abs(cosine(x1[i], x2)))
    #print(dis)
    return np.array(dis)

def getDistances(x1, x2, valid_idx = None, weights = None, use_test = None):
    # weights should be a list, indicating the weights distribution of cosine, L1 and L2 distance
    # distances between 4w train and 1w valid
    
    dir_name = './Distances'
    total_dis_name = dir_name + '/distances.npy'
    weights_filename = dir_name + '/weights.npy'
    distances_matrix = []
    num_test = x2.shape[0]

    if use_test == None:
        cos_dis_name = dir_name + '/cos_dis_valid' + str(valid_idx) + '.npy'
        l1_dis_name = dir_name + '/l1_dis_valid' + str(valid_idx) + '.npy'
        l2_dis_name = dir_name + '/l2_dis_valid' + str(valid_idx) + '.npy'
    else:
        cos_dis_name = dir_name + '/cos_dis_alldata.npy'
        l1_dis_name = dir_name + '/l1_dis_alldata.npy'
        l2_dis_name = dir_name + '/l2_dis_alldata.npy'


    # verify dir
    if not os.path.exists(dir_name):
        print("'./Distances' doesn't exist, create one")
        os.makedirs(dir_name)
    else:
        print("'./Distances' detected, go on")

    if os.path.exists(weights_filename):
        last_weights = np.load(weights_filename)
        if last_weights == np.array(weights):
            return np.load(total_dis_name)
        else:
            print('weights matrix has been changed, recalculate total distances')
    else:
        print("weights npy file doesn't exist, create one")
    np.save(weights_filename, np.array(weights))

    
    # get cos, L1 and L2
    if os.path.exists(cos_dis_name) and os.path.exists(l1_dis_name) and os.path.exists(l2_dis_name):
        print('%s detected, load npy data' %cos_dis_name)
        print('%s detected, load npy data' %l1_dis_name)
        print('%s detected, load npy data' %l2_dis_name)
        cos_distances = np.load(cos_dis_name)
        l1_distances = np.load(l1_dis_name)
        l2_distances = np.load(l2_dis_name)
    else:
        print('Start to create npy files for cosine, L1 and L2')
        cos_distances = []
        l1_distances = []
        l2_distances = []
        for i in range(num_test):
            print('%d of %d finished' %(i, num_test))
            cos_distances.append(cosDis(x1, x2[i]))
            l1_distances.append(LmNorm(x1, x2[i], 1))
            l2_distances.append(LmNorm(x1, x2[i], 2))
        np.save(cos_dis_name, np.array(cos_distances))
        print('cosine finished')
        np.save(l1_dis_name, np.array(l1_distances))
        print('L1 finished')
        np.save(l2_dis_name, np.array(l2_distances))
        print('L2 finished')
        print('Finish all for npy data\n!')

    # get total distances
    print('Start total distances')
    for i in range(num_test):
        distances_matrix.append(weights[0]*cos_distances[i] + weights[1]*l1_distances[i] + weights[2]*l2_distances[i])
    np.save(total_dis_name, np.array(distances_matrix))
    print('Finish total distances')
    return np.array(distances_matrix)

class KNearestNeighbor:
    def __init__(self):
        pass
    
    def train(self, x, y):
        self.xtr = x
        self.ytr = y
        
    def predict(self, x, k = None, valid_idx = None):
        if k == None:
            k = 10
        else:
            self.value_k = k
        if valid_idx == None:
            self.valid_idx = 5
        else:
            self.valid_idx = valid_idx
        
        print('\nStart to process\n')
        num_test = x.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        self.dis_weights = [0, 0, 1]
        distances_matrix = getDistances(self.xtr, x, valid_idx = valid_idx, weights = self.dis_weights, use_test = None)
        for i in range(num_test):
            #distances = cosDis(self.xtr, x[i])
            #distances = LmNorm(self.xtr, x[i], 2)
            #distances = np.sum(np.abs(self.xtr - x[i]), axis=1)
            print(distances_matrix[i])
            indexs = np.argsort(distances_matrix[i]) #对index排序
            closestK = self.ytr[indexs[:k]] #取距离最小的K个点的标签值
            print('closestK is ', closestK, '\n')
            count = np.bincount(closestK) #获取各类的得票数
            Ypred[i] = np.argmax(count) #找出得票数最多的一个
            if (i+1) % 10 == 0:
               print('now: %d/%d, Ypred[%d] = %d\r' % (i+1, num_test, i, Ypred[i]))
        return Ypred
    
    def evaluate(self, Ypred, y):
        num_test = len(y)
        num_correct = np.sum(Ypred == y)
        accuracy = float(num_correct) / num_test
        print("[cos, l1, l2] = ", self.dis_weights, "With k = %d, %d / %d correct => accuracy: %.2f %%" %(self.value_k, num_correct, num_test, accuracy*100)) 
        return accuracy

if __name__ == "__main__":
    valid_idx = 5
    k = 10

    x_train, y_train, x_valid, y_valid, x_test, y_test = loadAll(dataDir, valid_idx)

    classifier = KNearestNeighbor()
    classifier.train(x_train, y_train)
    result = classifier.predict(x_valid, k, valid_idx)
    classifier.evaluate(result, y_valid)
