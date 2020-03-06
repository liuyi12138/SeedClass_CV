#!/bin/env python3

import numpy as np

class softmax_classifier(object):
    def __init__(self, net_layer_shapes, use_biases=False):
        self._is_properly_init = True
        self._use_biases = use_biases
        # set up the weights and biases
        if len(net_layer_shapes) <= 2:
            print("Wrong network description!")
            self._is_properly_init = False
            return None
        else:
            self._input_shape = net_layer_shapes[0]
            self._output_shape = net_layer_shapes[-1]
            self._net_weights = []
            if self._use_biases: self._net_biases = []

            weights_num = len(net_layer_shapes) - 1
            for i in range(1, weights_num):
                weight_range = np.sqrt(6 / (net_layer_shapes[i - 1] + net_layer_shapes[i]))
                self._net_weights.append(np.random.uniform(-weight_range, weight_range, (net_layer_shapes[i - 1], net_layer_shapes[i])))  # Xavier initialization reference https://zhuanlan.zhihu.com/p/76602243

                if self._use_biases:
                    bias_range = np.sqrt(1 / net_layer_shapes[i])
                    self._net_biases.append(np.random.uniform(-bias_range, bias_range, net_layer_shapes[i]))

        # set up the partial derivatives
    def _back_propagate(self, input, tag):
        """
        store the propagation result accumulatively into a temporary updating parameters structure
        :param input: the input image
        :param tag: the tag should be the correct class index
        """
        if self._is_properly_init:
            # check shape
            if self._input_shape == len(input):
                prob_results, _index, _results = self.predict(input, is_return_inter_values=True)

                # output layer partial derivatives
                output_layer_partial_derivs = np.zeros(self._output_shape)
                for i in len(output_layer_partial_derivs):
                    output_layer_partial_derivs[i] = prob_results[i]-1 if i == tag else prob_results[i]

                # todo: weights and biases back propagation
                pass

                # store the propations result
                pass

    def _apply_propagation(self):
        """
        apply the temporary parameters updates to the weights and biases
        """
        if self._is_properly_init:
            # update all the propagation
            pass

    def predict(self, input, is_return_inter_values=True):
        if self._is_properly_init:
            inter_value = input
            inter_results = []

            if self._use_biases:
                for idx, weight_mat in enumerate(self._net_weights):
                    inter_value = inter_value.dot(weight_mat)+self._use_biases[idx]
                    if is_return_inter_values:
                        inter_results.append(inter_value.copy())
            else:
                for weight_mat in self._net_weights:
                    inter_value = inter_value.dot(weight_mat)
                    if is_return_inter_values:
                        inter_results.append(inter_value.copy())

            # softmax function result and return
            inter_value = np.exp(inter_value)
            prob_results = inter_value / np.sum(inter_value)
            pred_index = np.argmax(prob_results)
            return prob_results, pred_index, inter_results


    def batch_train(self, batch_data, tags):
        """
        :param batch_data:
        :param tags:
        """
        if self._is_properly_init:
            for idx, input in enumerate(batch_data):
                self._back_propagate(input, tags[idx])
            self._apply_propagation()

# 数据集分batch的职责由外部实现

if __name__ == "__main__":
    # tests are written below
    from dataProcess import load_one

    clsfir = softmax_classifier