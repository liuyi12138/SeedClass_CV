#!/bin/env python3

from dataProcess import loadAll
from KNN import KNearestNeighbor
import numpy as np

if __name__ == "__main__":
    valid_idx = 5
    x_train, y_train, x_valid, y_valid, x_test, y_test = loadAll(valid_idx)

    classifier = KNearestNeighbor()
    classifier.train(x_train, y_train)
    for k in range(1, 101):
        result = classifier.predict(data_test=x_train, k=k, valid_idx=valid_idx)
        classifier.evaluate(result, y_valid[:1000])
