#!/bin/env python3

from dataProcess import loadAll
from KNN import KNearestNeighbor
import numpy as np

if __name__ == "__main__":
    valid_idx = 5
    x_train, y_train, x_valid, y_valid, x_test, y_test = loadAll(valid_idx)

    classifier = KNearestNeighbor()
    classifier.train(xtr_new, y_train)
    for k in range(1, 101):
        result = classifier.predict(x=xva_new[:1000], k=k, valid_idx=valid_idx, Optimizer=opt)
        classifier.evaluate(result, y_valid[:1000])
