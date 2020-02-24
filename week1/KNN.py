import os
import numpy as np
from dataProcess import loadAll, pca
from configTemplate import dataDir
from scipy.spatial.distance import cosine


def LmNormMetric(norm_argument):
    """
    This function returns a function, set the norm_argument to get the designated metric function
    0 for cos
    1 for 1-norm
    2 for 2-norm
    ...
    """
    if norm_argument == 0:
        def cosDis(labeled_points, point_to_calc):
            """
            Note that the power of 1/m is omitted for better performance
            labeld_points: should be a matrix
            point_to_calc: ought to be vector
            """
            dis_list = []
            for i in range(labeled_points.shape[0]):
                dis_list.append(cosine(labeled_points[i], point_to_calc))
            return np.array(dis_list)

        return cosDis
    else:
        def LmNorm(labeled_points, point_to_calc):
            """
            Note that the power of 1/m is omitted for better performance
            labeld_points: should be a matrix
            point_to_calc: ought to be vector
            """
            dis_list = []
            for i in range(labeled_points.shape[0]):
                distance = np.sum(np.abs(labeled_points - point_to_calc)) if norm_argument == 1 \
                    else np.sum((labeled_points - point_to_calc) ** norm_argument)
                dis_list.append(distance)
            return np.array(dis_list)

        return LmNorm


class NearestNeighbor:
    def train(self, data, labels):
        self.data_train = data
        self.labels_train = labels

    def predict(self, data_to_pred):
        num_data = data_to_pred.shape[0]
        pred_results = np.zeros(num_data, dtype=self.labels_train.dtype)
        for i in range(num_data):
            distances = np.sum(np.abs(self.data_train - data_to_pred[i]), axis=1)
            min_index = np.argmin(distances)
            pred_results[i] = self.labels_train[min_index]
            if (i + 1) % 10 == 0:
                print('now: %d/%d' % (i + 1, num_data))
        return pred_results


def getDistances(x1, x2, metric, name_tag):
    # Parameters Explanation:
    #   @x1, x2:        numpy 2D-matrixes, x1 is train, x2 is valid
    #   @metric:        this metric will be called to calculate the distance
    #   @name_tag:      name_tag will be used to name the files
    dump_path = 'distances_' + str(name_tag) + '.npy'  # total_dis = weights_matrix x distances_matrix

    distances_matrix = []
    num_test = x2.shape[0]

    # get cos, L1 and L2
    if os.path.exists(dump_path):
        print('%s detected, load npy data' % dump_path)
        distances_matrix = np.load(dump_path)
    else:
        print('Start to create npy!')
        for i in range(num_test):
            if (i + 1) % 100 == 0:
                print('%d of %d finished' % ((i + 1), num_test))
            distances_matrix.append(metric(x1, x2[i]))
        np.save(dump_path, np.array(distances_matrix))
        print('L2 finished')

    return np.array(distances_matrix)


class KNearestNeighbor:
    def train(self, x, y):
        self.xtr = x
        self.ytr = y

    def predict(self, x, k=10, metric=LmNormMetric(1)):
        self.value_k = k

        print('\nStart to process\n')
        num_test = x.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        self.dis_weights = [1, 0, 0]
        distances_matrix = getDistances(self.xtr, x, metric=metric, name_tag=k)
        for i in range(num_test):
            indexs = np.argsort(distances_matrix[i])  # 对index排序
            closestK = self.ytr[indexs[:k]]  # 取距离最小的K个点的标签值
            count = np.bincount(closestK)  # 获取各类的得票数
            Ypred[i] = np.argmax(count)  # 找出得票数最多的一个
        return Ypred

    def predict_2Class(self, test_data, k, m, f):
        """
        k: the KNN argument
        m: metric
        f: file name tag
        """
        num_test = test_data.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        distances = getDistances(self.xtr, test_data, LmNormMetric(m), str(m) + str(f))

        for i in range(num_test):
            indexs = np.argsort(distances[i])  # 对index排序
            closestK = self.ytr[indexs[:k]]  # 取距离最小的K个点
            count = np.bincount(closestK)  # 获取各类的得票数
            Ypred[i] = np.argmax(count)  # 找出得票数最多的一个
        return Ypred

    def predict_5Class(self, test_data, k, m, f, result_2Class):
        """
        k: the KNN argument
        m: metric
        f: file name tag
        """
        num_test = test_data.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        distances = getDistances(self.xtr, test_data, LmNormMetric(m), str(m) + str(f))

        for i in range(num_test):
            indexs = np.argsort(distances[i])  # 对index排序
            allDis = self.ytr[indexs]  # 获取到所有数据
            deleteINdex = []
            if result_2Class[i] == 1:
                for j in range(1000):
                    if (allDis[j] not in [0, 1, 8, 9]):
                        deleteINdex.append(j)
            else:
                for j in range(1000):
                    if (allDis[j] in [0, 1, 8, 9]):
                        deleteINdex.append(j)
            allDis = np.delete(allDis, deleteINdex)
            closestK = allDis[:k]  # 取前K个点
            count = np.bincount(closestK)  # 获取各类的得票数
            Ypred[i] = np.argmax(count)  # 找出得票数最多的一个
        return Ypred

    def evaluate(self, Ypred, y):
        num_test = len(y)
        num_correct = np.sum(Ypred == y)
        accuracy = float(num_correct) / num_test
        print("[cos, L1, L2] = ", self.dis_weights, "With k = %d, %d / %d correct => accuracy: %.2f %%" % (
        self.value_k, num_correct, num_test, accuracy * 100))
        return accuracy


if __name__ == "__main__":
    valid_idx = 5
    x_train, y_train, x_valid, y_valid, x_test, y_test = loadAll(valid_idx)

    x_valid = np.load(dataDir + '/x.npy').reshape(1000, 3072)
    y_valid = np.load(dataDir + '/y.npy').reshape(1000, )

    xtr_new, xva_new = pca(x_train, x_valid, n_components=30)
    print(xva_new.shape)

    classifier = KNearestNeighbor()
    classifier.train(xtr_new, y_train)
    for k in range(1, 101):
        result = classifier.predict(x=xva_new[:1000], k=k)
        classifier.evaluate(result, y_valid[:1000])
