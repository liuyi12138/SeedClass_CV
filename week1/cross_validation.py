import numpy as np
import random
from KNN import KNearestNeighbor
from dataProcess import loadAll
from configTemplate import dataDir

def cross_valid(k = None, m = None):
    # k, m are hyperparameters
    # cls_list = list()
    for valid_idx in range(1, 6):
        x_train, y_train, x_valid, y_valid, x_test, y_test = loadAll(dataDir, valid_idx = valid_idx)
        rd_start = random.randint(0, ((5-(valid_idx != 0)) * 10000) - 1)
        xtr = x_train[rd_start:10000]
        ytr = y_train[rd_start:10000]
        # cls_list.append(KNearestNeighbor())
        # cls_list[valid_idx-1].train(xtr, ytr)
        classifier = KNearestNeighbor()
        classifier.train(xtr, ytr)
        Ypred = classifier.predict(x_valid, k = k, m = m)
        print('\nvalid_idx = %d' %valid_idx)
        classifier.evaluate(Ypred, y_valid)

if __name__ == '__main__':
    cross_valid(k = 10, m = 2)