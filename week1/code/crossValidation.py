import random
from KNN import KNearestNeighbor
from dataProcess import loadAll


def cross_valid(k=None, m=None):
    """
    k: k the hyperparameter of KNN
    m: metric used to decide the distances between the images
    """
    # k, m are hyperparameters
    rd_start = random.randint(0, 3 * 10000 - 1)
    for valid_idx in range(1, 6):
        print('\nvalid_idx = %d' % valid_idx)
        data_train, labels_train, data_valid, labels_valid, data_test, labels_test = loadAll(valid_idx)
        xtr = data_train[rd_start:rd_start + 10000]
        ytr = labels_train[rd_start:rd_start + 10000]
        xva = data_valid[:1000]
        yva = labels_valid[:1000]

        classifier = KNearestNeighbor()
        classifier.train(xtr, ytr)
        Ypred = classifier.predict(xva, k=k, m=m)
        classifier.evaluate(Ypred, yva)


if __name__ == '__main__':
    cross_valid(k=10, m=2)
