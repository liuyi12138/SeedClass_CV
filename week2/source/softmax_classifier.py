#!/bin/env python3

import numpy as np

class softmax_classifier(object):
    def __init__(self, net_layer_shapes):
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
            self._pending_weights = []      # used to store those to-be-applied weights

            weights_num = len(net_layer_shapes)-1
            for i in range(1, weights_num+1):
                weight_range = np.sqrt(6 / (net_layer_shapes[i - 1] + net_layer_shapes[i]))
                self._net_weights.append(np.random.uniform(-weight_range, weight_range, (net_layer_shapes[i-1]+1, net_layer_shapes[i])))  # Xavier initialization reference https://zhuanlan.zhihu.com/p/76602243
                self._pending_weights.append(np.zeros((net_layer_shapes[i-1]+1, net_layer_shapes[i])))

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
                loss = -np.log(np.sum(one_hot_tag*prob_results))

                # w += (x.T).dot(p-one_hot)
                self._pending_weights[0] -= np.mat(np.concatenate((input,[1]), axis=0)).T.dot(np.mat(prob_results-one_hot_tag))*learning_rate
                return loss

    def _apply_propagation(self, division):
        """
        apply the temporary parameters updates to the weights and biases
        """
        if self._is_properly_init:
            # update all the propagation
            for i in range(len(self._net_weights)):
                self._net_weights[i] += self._pending_weights[i]/division
                self._pending_weights[i] *= 0      # reset the weights to zeros

    def predict(self, input, is_return_inter_values=False):
        if self._is_properly_init:
            inter_value = np.concatenate((input, [1]), axis=0)
            inter_results = []

            for weight_mat in self._net_weights:
                inter_value = inter_value.dot(weight_mat)
                if is_return_inter_values: inter_results.append(inter_value)

            # softmax function result and return
            inter_value = np.exp(inter_value)
            prob_results = inter_value / np.sum(inter_value)
            pred_index = np.argmax(prob_results)
            return prob_results, pred_index, inter_results

    def batch_train(self, batch_data, tags, learning_rate):
        """
        :param batch_data:
        :param tags:
        """
        batch_size = len(tags)
        total_loss = 0

        if self._is_properly_init:
            for idx, input in enumerate(batch_data):
                total_loss += self._back_propagate(input, tags[idx], learning_rate)
            self._apply_propagation(batch_size)
        total_loss /= batch_size
        return total_loss

# 数据集分batch的职责由外部实现

if __name__ == "__main__":
    # tests are written below
    clsfir = softmax_classifier((10, 10))
    assert clsfir._net_weights[0].shape == (11, 10)
    print("weight matrix shape:", clsfir._net_weights[0].shape)

    batch_data = np.eye(10)
    tags = list(range(10))
    print("batch_data: {}, tags: {}".format(batch_data, tags))

    for i in range(100):
        print("for batch_{}, loss is {} ".format(i, clsfir.batch_train(batch_data, tags, 10)))  # print loss when trainiing

    for i in range(10):
        print("right answer: {}, prediction: {}".format(i, clsfir.predict(batch_data[i])[1]))             # print prediction result

