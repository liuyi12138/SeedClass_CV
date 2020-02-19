import numpy as np
import random
from KNN import KNearestNeighbor
from dataProcess import loadAll
from configTemplate import dataDir

def cross_valid(k = None, m = None):
    # k, m are hyperparameters
    #cls_list = list()
    rd_start = random.randint(0, 3 * 10000 - 1)
    for valid_idx in range(1, 6):
        print('\nvalid_idx = %d' %valid_idx)
        x_train, y_train, x_valid, y_valid, x_test, y_test = loadAll(dataDir, valid_idx)
        xtr = x_train[rd_start:rd_start + 10000]
        ytr = y_train[rd_start:rd_start + 10000]
        xva = x_valid[:1000]
        yva = y_valid[:1000]
        #cls_list.append(KNearestNeighbor())
        #cls_list[valid_idx-1].train(xtr, ytr)
        classifier = KNearestNeighbor()
        classifier.train(xtr, ytr)
        Ypred = classifier.predict(xva, k = k, m = m)
        classifier.evaluate(Ypred, yva)

if __name__ == '__main__':
    cross_valid(k = 10, m = 2)