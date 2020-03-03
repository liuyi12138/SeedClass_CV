#!/bin/env python3

import numpy as np

class softmax_classifier(object):
    def __init__(self, net_layers, use_biases=False):
        self._is_properly_init = True
        self._use_biases = use_biases
        # set up the weights and biases
        if len(net_layers) <= 2:
            print("Wrong network description!")
            self._is_properly_init = False
            return None
        else:
            self._input = None
            self._output = None
            self._net_weights = []
            if self._use_biases: self._net_biases = []

            weights_num = len(net_layers) - 1
            for i in range(1, weights_num):
                self._net_weights.append(np.zeros(net_layers[i-1]), np.zeros(net_layers[i]))
                if self._use_biases: self._net_biases.append(np.zeros(net_layers[i]))

        # set up the partial derivatives


        return None
