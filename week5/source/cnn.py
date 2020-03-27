import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers

def unpickle(filename):
    """
    data_dict: a object consists of data and labels
    """
    import pickle
    with open(filename, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def loadOne(filename):
    # load single batch of cifar dataset
    datadict = unpickle(filename)
    data = datadict[b'data']
    labels = np.array(datadict[b'labels'])
    return data, labels

def normalizationImage(data):
    data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    data = np.array(data)/255
    data = data - 0.5
    return data

def OneHogY(data):
    one_hot_data = []
    for i in range(len(data)):
        one_hot_temp = np.zeros(10)
        one_hot_temp[data[i]] = 1
        one_hot_data.append(one_hot_temp)
    one_hot_data = np.array(one_hot_data)
    return one_hot_data

for i in range(1, 6):
    if i == 1:
        x_train, y_train = loadOne("../../../cifar-10-batches-py/data_batch_1")
    else:
        x_temp, y_temp = loadOne("../../../cifar-10-batches-py/data_batch_" + str(i))
        x_train = np.concatenate((x_train, x_temp), axis=0)
        y_train = np.concatenate((y_train, y_temp), axis=0)
x_test, y_test = loadOne("../../../cifar-10-batches-py/test_batch")

x_train = normalizationImage(x_train)
x_test = normalizationImage(x_test)

y_train = OneHogY(y_train)
y_test = OneHogY(y_test)

# 网络模型参考 https://geektutu.com/post/tf2doc-cnn-cifar10.html
# Conv2D 网络层 MaxPooling2D 池化层 Dense 全连接层
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3))) #输出为[28,28，32]
model.add(layers.MaxPooling2D((2, 2))) #输出为[14,14，32]
model.add(layers.Conv2D(64, (5, 5), activation='relu')) #输出为[10,10，64]
model.add(layers.MaxPooling2D((2, 2))) #输出为[5,5，64]
model.add(layers.Conv2D(64, (5, 5), activation='relu')) #输出为[1,1，64]
model.add(layers.Flatten()) #输出为[64]
model.add(layers.Dense(32, activation='relu')) #输出为[32]
model.add(layers.Dense(10, activation='softmax')) #输出为[10]
model.summary()

# 优化器 目标函数等参考 https://keras-cn.readthedocs.io/en/latest/legacy/other/optimizers/
sgd = optimizers.SGD(lr=2e-4, decay=0.9, momentum=0.0, nesterov=False)
adam = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
 #优化器sgd 目标函数为softmax+对数损失
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20)

test_loss, test_acc = model.evaluate(x_test, y_test)
test_acc