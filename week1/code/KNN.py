import os
import numpy as np
from dataProcess import loadAll, pca
from scipy.spatial.distance import cosine


def MetricGenerator(metric_type):
    """
    This function returns a function, set the norm_argument to get the designated metric function
    0 for cos
    1 for 1-norm
    2 for 2-norm
    ...
    """
    if metric_type == 0:
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
                distance = np.sum(np.abs(labeled_points - point_to_calc)) if metric_type == 1 \
                    else np.sum((labeled_points - point_to_calc) ** metric_type)
                dis_list.append(distance)
            return np.array(dis_list)

        return LmNorm

def getDistances(data_train, data_test, metric, name_tag):
    # Parameters Explanation:
    #   @x1, x2:        numpy 2D-matrixes, x1 is train, x2 is valid
    #   @metric:        this metric will be called to calculate the distance
    #   @name_tag:      name_tag will be used to name the files
    dump_path = "distances_{}_.npy".format(str(name_tag))  # total_dis = weights_matrix x distances_matrix

    distances_matrix = []
    num_test = data_test.shape[0]

    if os.path.exists(dump_path):
        print('%s detected, load npy data' % dump_path)
        distances_matrix = np.load(dump_path)
    else:
        print('Start to create npy!')
        for i in range(num_test):
            if (i + 1) % 100 == 0:
                print('%d of %d finished' % ((i + 1), num_test))
            distances_matrix.append(metric(data_train, data_test[i]))
        np.save(dump_path, np.array(distances_matrix))
        print("distance data saved at {}".format(dump_path))

    return np.array(distances_matrix)


class KNearestNeighbor:
    def train(self, data_train, labels_train):
        self.data_train = data_train
        self.labels_train = labels_train

    def predict(self, data_test, k=10, metric=MetricGenerator(1)):
        self.value_k = k

        print('\nStart to process\n')
        num_test = data_test.shape[0]
        pred_results = np.zeros(num_test, dtype=self.labels_train.dtype)

        distances_matrix = getDistances(self.data_train, data_test, metric=metric, name_tag=k)
        for i in range(num_test):
            indexs = np.argsort(distances_matrix[i])   # 对index排序
            closest_k = self.labels_train[indexs[:k]]  # 取距离最小的K个点的标签值
            votes = np.bincount(closest_k)             # 获取各类的得票数
            pred_results[i] = np.argmax(votes)         # 找出得票数最多的一个
        return pred_results

    def predict_2Class(self, data_test, k, metric_type, name_tag):
        num_test = data_test.shape[0]
        pred_results = np.zeros(num_test, dtype=self.labels_train.dtype)
        distances = getDistances(self.data_train, data_test, MetricGenerator(metric_type), str(metric_type) + str(name_tag))

        for i in range(num_test):
            indexs = np.argsort(distances[i])          # 对index排序
            closest_k = self.labels_train[indexs[:k]]  # 取距离最小的K个点
            votes = np.bincount(closest_k)             # 获取各类的得票数
            pred_results[i] = np.argmax(votes)         # 找出得票数最多的一个
        return pred_results

    def predict_5Class(self, data_test, k, metric_type, name_tag, result_2Class):
        class_one = [0, 1, 8, 9]
        num_test = data_test.shape[0]
        pred_results = np.zeros(num_test, dtype=self.labels_train.dtype)
        distances = getDistances(self.data_train, data_test, MetricGenerator(metric_type), str(metric_type) + str(name_tag))

        for i in range(num_test):
            indexs = np.argsort(distances[i])   # 对index排序
            labels = self.labels_train[indexs]  # 获取到所有数据
            ind_del = []
            if result_2Class[i] == 1:
                for j in range(1000):
                    if (labels[j] not in class_one):
                        ind_del.append(j)
            else:
                for j in range(1000):
                    if (labels[j] in class_one):
                        ind_del.append(j)
            labels = np.delete(labels, ind_del)
            closest_k = labels[:k]               # 取前K个点
            votes = np.bincount(closest_k)       # 获取各类的得票数
            pred_results[i] = np.argmax(votes)   # 找出得票数最多的一个
        return pred_results

    def evaluate(self, pred_results, labels_test):
        num_test = len(labels_test)
        num_correct = np.sum(pred_results == labels_test)
        accuracy = float(num_correct) / num_test
        print("With k = %d, %d / %d correct => accuracy: %.2f %%" % (self.value_k, num_correct, num_test, accuracy * 100))
        return accuracy


if __name__ == "__main__":
    valid_idx = 5
    data_train, labels_train, data_valid, labels_valid, _, _ = loadAll(valid_idx)

    data_train_pca, data_valid_pca = pca(data_train, data_valid, n_components=30)

    classifier = KNearestNeighbor()
    classifier.train(data_train_pca, labels_train)
    for k in range(1, 101):
        result = classifier.predict(data_test=data_valid_pca[:1000], k=k)
        classifier.evaluate(result, labels_valid[:1000])
