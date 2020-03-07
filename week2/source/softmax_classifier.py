#!/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class softmax_classifier(object):
    def __init__(self, net_layer_shapes, k, L):
        self._is_properly_init = True
        # set up the weights and biases
        if len(net_layer_shapes) < 2:
            print("Wrong network description!")
            self._is_properly_init = False
            return None
        else:
            self._input_shape = net_layer_shapes[0]
            self._output_shape = net_layer_shapes[-1]
            self._net_weights = []
            self._pending_weights = []  # used to store those to-be-applied weights
            self._k = k
            self._L = L

            weights_num = len(net_layer_shapes) - 1
            for i in range(1, weights_num + 1):
                weight_range = np.sqrt(6 / (net_layer_shapes[i - 1] + net_layer_shapes[i]))
                self._net_weights.append(np.random.uniform(-weight_range, weight_range, (net_layer_shapes[i - 1] + 1,
                                                                                         net_layer_shapes[
                                                                                             i])))  # Xavier initialization reference https://zhuanlan.zhihu.com/p/76602243
                self._pending_weights.append(np.zeros((net_layer_shapes[i - 1] + 1, net_layer_shapes[i])))

        # set up the partial derivatives

    def _back_propagate(self, input, tag, learning_rate):
        """
        store the propagation result accumulatively into a temporary updating parameters structure
        :param input: the input image
        :param tag: the tag should be the correct class index
        :return loss
        """
        if self._is_properly_init:
            # check shape
            if self._input_shape == len(input):
                prob_results, pred_index, inter_results = self.predict(input, is_return_inter_values=True)

                # output layer partial derivatives
                one_hot_tag = np.zeros(self._output_shape)
                one_hot_tag[tag] = 1
                loss = -np.log(np.sum(one_hot_tag * prob_results))

                # w += (x.T).dot(p-one_hot)
                current_derivative = np.mat(prob_results - one_hot_tag)
                for i in range(-1, -len(self._net_weights), -1): # totally len(self._net_weights)-1
                    self._pending_weights[i] -= np.mat(np.concatenate((inter_results[i-1], [1]))).T.dot(
                        current_derivative
                    ) * learning_rate
                    current_derivative = current_derivative.dot(self._net_weights[i][:-1,:].T)
                self._pending_weights[0] -= np.mat(np.concatenate((input, [1]))).T.dot(current_derivative) * learning_rate

                    # next
                return loss

    def _apply_propagation(self, division, learning_rate):
        """
        apply the temporary parameters updates to the weights and biases
        :param division
        """
        if self._is_properly_init:
            # update all the propagation
            for i in range(len(self._net_weights)):
                if (self._L == 1):
                    self._net_weights[i] -= self._k * np.sign(self._net_weights[i]) * learning_rate / len(
                        self._net_weights[i])  # L1
                if (self._L == 2):
                    self._net_weights[i] -= self._k * self._net_weights[i] * learning_rate / len(
                        self._net_weights[i])  # L2

                self._net_weights[i] += self._pending_weights[i] / division
                self._pending_weights[i] *= 0  # reset the weights to zeros

    def predict(self, input, is_return_inter_values=False):
        """

        :param input: data input
        :param is_return_inter_values: when True, this function returns the intermediate values for back propagation
        :return inter_results stores the intermediate values of those layers except input
        """
        if self._is_properly_init:
            inter_value = input
            inter_results = []

            for weight_mat in self._net_weights:
                inter_value = np.concatenate((inter_value,[1])).dot(weight_mat)
                if is_return_inter_values: inter_results.append(inter_value)

            # softmax function result and return
            inter_value = np.exp(inter_value)
            prob_results = inter_value / np.sum(inter_value)
            pred_index = np.argmax(prob_results)
            return prob_results, pred_index, inter_results

    def batch_train(self, batch_data, tags, learning_rate):
        """
        :param batch_data:
        :param tags: the right answers of the batch, tags should be digits
        """
        batch_size = len(tags)
        if batch_data.shape[0] != batch_size:
            print("Wrong training data!")
            return None
        total_loss = 0

        if self._is_properly_init:
            for idx, input in enumerate(batch_data):
                total_loss += self._back_propagate(input, tags[idx], learning_rate)
            self._apply_propagation(batch_size, learning_rate)
        total_loss /= batch_size
        return total_loss


# 数据集分batch的职责由外部实现


def unpickle(filename):
    """
    data_dict: a object consists of data and labels
    """
    import pickle
    with open(filename, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def loadOne(filename):
    # load single batch of cifar dataset
    datadict = unpickle(filename)
    data = datadict[b'data']
    labels = np.array(datadict[b'labels'])
    return data, labels


def normalizationImage(data):
    data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    data = data.reshape(data.shape[0], 3072)
    return np.array(data) / 255


def pca(data_train, data_test, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data_train)
    data_train_pca = pca.transform(data_train)
    data_test_pca = pca.transform(data_test)
    return data_train_pca, data_test_pca


if __name__ == "__main__":
    # tests are written below
    if True:
        clsfir = softmax_classifier((10, 4, 10), 10, 10)
        assert clsfir._net_weights[0].shape == (11, 4)
        print("weight matrix shape:", clsfir._net_weights[0].shape)

        batch_data = np.eye(10)
        tags = list(range(10))
        print("batch_data: {}, tags: {}".format(batch_data, tags))

        for i in range(100):
            print("for batch_{}, loss is {} ".format(i,
                                                     clsfir.batch_train(batch_data, tags, 1)))  # print loss when trainiing

        for i in range(10):
            print("right answer: {}, prediction: {}".format(i, clsfir.predict(batch_data[i])[1]))  # print prediction result
    else:
        x_train,y_train = loadOne("D:/HUST/寒假课程资料/数字图像处理/课设/week1/cifar-10-batches-py/data_batch_1")
        x_test,y_test = loadOne("D:/HUST/寒假课程资料/数字图像处理/课设/week1/cifar-10-batches-py/data_batch_2")
        x_train = normalizationImage(x_train)
        x_test = normalizationImage(x_test)

        k = 10
        L = 1
        batch_size = 256
        learning_rate = 0.02
        epoch = 100

        clsfir = softmax_classifier((3072, 10),k,L)
        loss = []

        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        for i in range(epoch):
            n = 0
            lossAll = 0
            while(n + batch_size < len(x_train)):
                x_temp = x_train[n : n+batch_size]
                y_temp = y_train[n : n+batch_size]
                lossAll += clsfir.batch_train(x_temp, y_temp, learning_rate)
                n += batch_size
            loss_temp = float(lossAll)/((n/batch_size)+1)
            loss.append(loss_temp)
            print("for epoch{}, loss is {} ".format(i, loss_temp))

            permutation = np.random.permutation(y_train.shape[0])
            x_train = x_train[permutation]
            y_train = y_train[permutation]

        plt.plot(loss)
        plt.show()

        result = []
        for i in range(10000):
            predict = clsfir.predict(x_test[i])[1]
            result.append(predict)

        num_test= 10000
        num_correct = np.sum(result == y_test) #计算准确率
        accuracy = float(num_correct) / num_test
        print("acc: %f" % accuracy)
