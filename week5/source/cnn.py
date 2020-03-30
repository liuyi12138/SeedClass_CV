import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers

# 屏蔽tf的通知信息、警告信息 (如果设置为3，则还屏蔽报错信息)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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


for i in range(1, 6):
    if i == 1:
        x_train, y_train = loadOne("../../../cifar-10-batches-py/data_batch_1")
    else:
        x_temp, y_temp = loadOne("../../../cifar-10-batches-py/data_batch_" + str(i))
        x_train = np.concatenate((x_train, x_temp), axis=0)
        y_train = np.concatenate((y_train, y_temp), axis=0)
x_test, y_test = loadOne("../../../cifar-10-batches-py/test_batch")

x_train = x_train.reshape(x_train.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
x_test = x_test.reshape(x_test.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#数据生成器
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.16,
    height_shift_range=0.16,
    zoom_range=0.2,
    rescale=1./255,
    horizontal_flip=True)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


# 网络模型参考 https://geektutu.com/post/tf2doc-cnn-cifar10.html
# Conv2D 卷积层 MaxPooling2D 池化层 Dense 全连接层
model = models.Sequential()
model.add(layers.BatchNormalization(input_shape=(32, 32, 3), axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_uniform(seed=None)))
model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_uniform(seed=None)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_uniform(seed=None)))
model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_uniform(seed=None)))
model.add(layers.MaxPooling2D((2, 2))) 

model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_uniform(seed=None)))
model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_uniform(seed=None)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_uniform(seed=None)))
model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_uniform(seed=None)))
model.add(layers.MaxPooling2D((2, 2))) 

model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add(layers.Flatten())
model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(layers.Dense(512, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=None)))
model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(layers.Dense(10, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=None)))
model.summary()

keras.utils.plot_model(model, show_shapes = True, show_layer_names = True)

# 优化器 目标函数等参考 https://keras-cn.readthedocs.io/en/latest/legacy/other/optimizers/
sgd = optimizers.SGD(lr=0.0001, decay=0.93, momentum=0.3, nesterov=False)
adamMax = keras.optimizers.Adamax(lr=0.0015, beta_1=0.7, beta_2=0.998, epsilon=1e-08)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
m_batch_size = 32
 #优化器adam 目标函数为softmax+对数损失
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit_generator(train_datagen.flow(x_train,y_train,batch_size=m_batch_size),steps_per_epoch=int(len(x_train)/m_batch_size), epochs=20)

test_loss, test_acc = model.evaluate_generator(test_datagen.flow(x_test, y_test))
test_acc