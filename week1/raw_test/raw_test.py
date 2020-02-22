#!/bin/env python3

from dataProcess import loadAll, dataDir
from KNN import Optimizer, KNearestNeighbor
import numpy as np

valid_idx = 5
x_train, y_train, x_valid, y_valid, x_test, y_test = loadAll(valid_idx)

x_valid = np.load(dataDir + '/x.npy').reshape(1000, 3072)
y_valid = np.load(dataDir + '/y.npy').reshape(1000, )

opt = Optimizer()
opt.generate(opt_type='PCA', opt_value=30)
opt.setWeights([1, 0, 0])

xtr_new, xva_new = pca(x_train, x_valid, n_components=opt.opt_value)
print(xva_new.shape)

classifier = KNearestNeighbor()
classifier.train(xtr_new, y_train)
for k in range(1, 101):
    result = classifier.predict(x=xva_new[:1000], k=k, valid_idx=valid_idx, Optimizer=opt)
    classifier.evaluate(result, y_valid[:1000])
