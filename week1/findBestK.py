import random
from KNN import KNearestNeighbor
from dataProcess import loadAll, dataDir

def findBestK():
    # k_list = [1, 2, 5, 10, 15, 30, 50, 100]
    x_train, y_train, x_valid, y_valid, x_test, y_test = loadAll(valid_idx = 5)
    classifier = KNearestNeighbor()

    # Select 10000 train data randomly
    rd_start = random.randint(0, 3 * 10000 - 1)
    xtr = x_train[rd_start:rd_start + 10000]
    ytr = y_train[rd_start:rd_start + 10000]
    xva = x_valid[:2000]
    yva = y_valid[:2000]

    # acc lists for plot
    m1_acc = [0]
    m2_acc = [0]
    
    classifier.train(xtr, ytr)
    for k in range(1, 101):
        Ypred_1 = classifier.predict(xva, k, m = 1)
        m1_acc.append(classifier.evaluate(Ypred_1, yva))
        Ypred_2 = classifier.predict(xva, k, m = 2)
        m2_acc.append(classifier.evaluate(Ypred_2, yva))
    # plotAcc()

if __name__ == '__main__':
    findBestK()