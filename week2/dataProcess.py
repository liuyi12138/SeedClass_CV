import numpy as np

#对list各元素取对数后标准化
def normalization(data):
    data = np.exp(data)
    _range = np.sum(data)
    return (data) / _range