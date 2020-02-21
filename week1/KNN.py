import numpy as np
from dataProcess import loadData, testDataDir, trainDataDir
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


def getDistances(x1, x2, valid_idx=None, weights=None, data_type=None, value=None):
    # Parameters Explaination:
    #   @x1, x2:        numpy 2D-matrixs, x1 is train, x2 is valid
    #   @valid_idx:     valid index from 1-5
    #   @weights:       A list indicating the weights distribution of cosine, L1 and L2 distance
    #   @data_type:     1-5, 1 for raw, 2 for Sample Grey, 3 for PCA, 4 for HOG, 5 for GreyHOG

    dir_list = ['./Dis-Raw', './Dis-SampleGrey', './Dis-PCA', './Dis-HOG', './Dis-GreyHOG']
    dir_name = dir_list[data_type - 1]
    total_dis = '/distances.npy'  # total_dis = weights_matrix x distances_matrix
    [cos_dis, l1_dis, l2_dis] = ['/cos_dis_' + str(value) + '.npy', '/l1_dis_' + str(value) + '.npy',
                                 '/l2_dis_' + str(value) + '.npy']
    weights_file = '/weights.npy'

    distances_matrix = []
    num_test = x2.shape[0]

    # npy file path of cos, l1 and l2
    cos_path = dir_name + cos_dis
    l1_path = dir_name + l1_dis
    l2_path = dir_name + l2_dis
    total_path = dir_name + total_dis

    # verify dir
    if not os.path.exists(dir_name):
        print("'./Distances' doesn't exist, create one")
        os.makedirs(dir_name)
    else:
        print("'./Distances' detected, go on")

    if os.path.exists(dir_name + weights_file) and os.path.exists(total_path):
        last_weights = np.load(dir_name + weights_file)
        if (last_weights == np.array(weights)).all() == True:
            return np.load(total_path)
        else:
            print('weights matrix has been changed, recalculate total distances')
    else:
        print("weights or total_dis npy file doesn't exist, create one")
    np.save(dir_name + weights_file, np.array(weights))

    # get cos, L1 and L2
    if os.path.exists(cos_path) and os.path.exists(l1_path) and os.path.exists(l2_path):
        print('%s detected, load npy data' % cos_path)
        print('%s detected, load npy data' % l1_path)
        print('%s detected, load npy data' % l2_path)
        cos_distances = np.load(cos_path)
        l1_distances = np.load(l1_path)
        l2_distances = np.load(l2_path)
    else:
        print('Start to create npy files for cosine, L1 and L2')
        cos_distances = []
        l1_distances = []
        l2_distances = []
        for i in range(num_test):
            if (i + 1) % 100 == 0:
                print('%d of %d finished' % ((i + 1), num_test))
            cos_distances.append(cosDis(x1, x2[i]))
            l1_distances.append(LmNorm(x1, x2[i], 1))
            l2_distances.append(LmNorm(x1, x2[i], 2))
        np.save(cos_path, np.array(cos_distances))
        print('cosine finished')
        np.save(l1_path, np.array(l1_distances))
        print('L1 finished')
        np.save(l2_path, np.array(l2_distances))
        print('L2 finished')

    # get total distances
    print('Start total distances')
    for i in range(num_test):
        distances_matrix.append(
            weights[0] * cos_distances[i] + weights[1] * l1_distances[i] + weights[2] * l2_distances[i])
    np.save(total_path, np.array(distances_matrix))
    print('Finish total distances')
    return np.array(distances_matrix)


class Optimizer:
    def __init__(self):
        self.dict = {'Raw': 1, 'SampleGrey': 2, 'PCA': 3, 'HOG': 4, 'GreyHOG': 5}
        self.weights = [1, 0, 0]

    def generate(self, opt_type=None, opt_value=None):
        self.opt_type = self.dict[opt_type]
        self.opt_value = opt_value

    def setWeights(self, weights):
        self.weights = weights


class KNearestNeighbor:
    def __init__(self):
        return None
    
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
            distances = cosDis(self.xtr, x[i])
            #distances = LmNorm(self.xtr, x[i], m)
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
    x_train, y_train = loadData(trainDataDir)
    x_test, y_test = loadData(testDataDir)
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
