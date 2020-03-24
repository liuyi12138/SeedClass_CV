import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from skimage.feature import hog
import time

class softmax_classifier(object):
    def __init__(self, net_layer_shapes = None, norm_ratio = None, norm_method = None, activation = "relu", parameter_initializer = "He", optimizer = "BGD"):
        self._is_properly_init = True
        # set up the weights and biases
        if len(net_layer_shapes) < 2:
            print("Wrong network description!")
            self._is_properly_init = False
            return None
        else:
            self._input_shape = net_layer_shapes[0]
            self._output_shape = net_layer_shapes[-1]
            self._net_weights = []
            self._pending_weights = []  # used to store those to-be-applied weights
            self._norm_ratio = norm_ratio
            self._norm_method = norm_method
            self._activation_method = activation
            self._optimizer = optimizer

            # normal regulation
            self._norm_func = lambda x: 0
            self._norm_derivative = lambda x: 0
            if self._norm_method == 1:
                self._norm_func = lambda x: np.sum(np.abs(x))
                self._norm_derivative = lambda x: self._norm_ratio * np.sign(x)
            elif self._norm_method == 2:
                self._norm_func = lambda x: np.sum(x ** 2)
                self._norm_derivative = lambda x: self._norm_ratio * 2 * x

            # activation function setup
            self._act_func = lambda x: x
            if self._activation_method == "relu":
                self._act_func = lambda x: np.multiply(x, (x > 0))
            if self._activation_method == "tanh":
                self._act_func = lambda x: np.tanh(x)
            if self._activation_method == "leaky_relu":
                self._act_func = LeakyRelu
            if self._activation_method == "elu":
                self._act_func = Elu

            self._t = 1        # 参数更新的次数
            if self._optimizer == "Momentum":
                self._m = []   # 保存上一次的动量矩阵，初始为空
                self._u = 0.9  # 动量因子
            elif self._optimizer == "Adam":
                self._n = []
                self._u = 0.9
                self._v = 0.999
                self._delta = 1e-8

            weights_num = len(net_layer_shapes) - 1
            for i in range(1, weights_num + 1):
                if parameter_initializer == "Xavier":
                    weight_range = np.sqrt(6 / (net_layer_shapes[i - 1] + net_layer_shapes[i]))
                    self._net_weights.append(np.random.uniform(-weight_range, weight_range, (net_layer_shapes[i - 1] + 1,
                                                                                            net_layer_shapes[i])))  # Xavier initialization reference https://zhuanlan.zhihu.com/p/76602243
                elif parameter_initializer == "He":
                    self._net_weights.append(np.random.normal(0, 2/net_layer_shapes[i - 1], (net_layer_shapes[i - 1] + 1, net_layer_shapes[i])))
                self._pending_weights.append(np.zeros((net_layer_shapes[i - 1] + 1, net_layer_shapes[i])))

        # 权重矩阵初始化完毕之后转化为numpy array
        self._net_weights = np.array(self._net_weights)
        self._pending_weights = np.array(self._pending_weights)

    def _back_propagate(self, input, tag, learning_rate):
        """
        store the propagation result accumulatively into a temporary updating parameters structure
        :param input: the input image
        :param tag: the tag should be the correct class index
        :return loss
        """
        if self._is_properly_init:
            # check shape
            if self._input_shape == len(input):
                prob_results, pred_index, inter_results = self.predict(input, is_return_inter_values=True)

                # output layer partial derivatives
                one_hot_tag = np.zeros(self._output_shape)
                one_hot_tag[tag] = 1
                loss = -np.log(np.sum(one_hot_tag * prob_results))
                net_loss = loss
                
                # calculate loss with regulation
                for weights in self._net_weights:
                    loss += self._norm_ratio * self._norm_func(weights)      

                # activation function setup
                self._act_derivative = lambda y: 1
                if self._activation_method == "relu":
                    self._act_derivative = lambda y: (y > 0)
                elif self._activation_method == "tanh":
                    self._act_derivative = lambda y: 1 - np.multiply(np.tanh(y), np.tanh(y))
                    # self._act_derivative = lambda y: 1 - np.multiply(y, y)
                elif self._activation_method == "leaky_relu":
                    self._act_derivative = LeakyRelu_derivative
                elif self._activation_method == "elu":
                    self._act_derivative = Elu_derivative

                # w += (x.T).dot(p-one_hot)
                gradient_mat = []
                current_derivative = np.mat(prob_results - one_hot_tag)
                for i in range(-1, -len(self._net_weights), -1):  # totally len(self._net_weights)-1
                    gradient = np.mat(np.concatenate((inter_results[i - 1], np.array([1])))).T.dot(current_derivative) + self._norm_derivative(self._net_weights[i])
                    gradient_mat.append(gradient) 
                    # derivative of results of activation function, so the biases is ignored here
                    current_derivative = current_derivative.dot(self._net_weights[i][:-1, :].T)
                    # derivative of activation function, given function results
                    current_derivative = np.multiply(self._act_derivative(np.array(inter_results[i - 1])), current_derivative)

                gradient = np.mat(np.concatenate((input, [1]))).T.dot(current_derivative) + self._norm_derivative(self._net_weights[0])
                gradient_mat.append(gradient)
                gradient_mat = np.array(gradient_mat)[::-1] # append时是倒序，现在修正
                
                if self._optimizer == "Momentum":
                    for i in range(len(gradient_mat)):
                        if len(self._m) != len(gradient_mat):
                            self._m.append(gradient_mat[i])
                        else:
                            self._m[i] = gradient_mat[i] + self._u * self._m[i]
                            gradient_mat[i] = self._m[i]
                elif self._optimizer == "Adam":
                    for i in range(len(gradient_mat)):
                        if len(self._m) != len(gradient_mat):
                            self._m.append((1 - self._u) * gradient_mat[i])
                            self._n.append((1 - self._v) * (gradient_mat[i] ** 2))
                        else:
                            self._m[i] = (1 - self._u) * gradient_mat[i] + self._u * self._m[i]
                            m_repaired = self._m[i] / (1 - self._u ** self._t)
                            self._n[i] = (1 - self._v) * (gradient_mat[i] ** 2) + self._v * self._n[i]
                            n_repaired = self._n[i] / (1 - self._v ** self._t)

                            gradient_mat[i] = m_repaired / (np.sqrt(n_repaired) + self._delta)
                
                self._pending_weights -= gradient_mat * learning_rate
                # self._pending_weights[0] -= gradient * learning_rate

                # next
                return loss, net_loss

    def _apply_propagation(self, division, learning_rate):
        """
        apply the temporary parameters updates to the weights and biases
        :param division
        """
        # normal regularization
        # norm_func = None
        # if self._norm_method == 1:
        #     norm_func = lambda x: self._norm_ratio * np.sign(x) * learning_rate
        # elif self._norm_method == 2:
        #     norm_func = lambda x: self._norm_ratio * 2 * x * learning_rate

        if self._is_properly_init:
            # update all the propagation
            for i in range(len(self._net_weights)):
                # if norm_func:
                #     self._net_weights[i] -= norm_func(self._net_weights[i])

                self._net_weights[i] += self._pending_weights[i] / division
                self._pending_weights[i] *= 0  # reset the weights to zeros
            self._t += 1

    def predict(self, input, is_return_inter_values=False):
        """
        :param input: data input
        :param is_return_inter_values: when True, this function returns the intermediate values for back propagation
        :return inter_results stores the intermediate values of those layers except input
        """
        if self._is_properly_init:
            inter_value = input
            # batch_size = len(inter_value)
            inter_results = []

            for idx, weight_mat in enumerate(self._net_weights):
                inter_value = np.concatenate((inter_value, [1])).dot(weight_mat)

                if idx != len(self._net_weights) - 1:
                    inter_value = self._act_func(inter_value)
                if is_return_inter_values: 
                    inter_results.append(inter_value)  # the inter_values are results of activation function

            # softmax function result and return
            inter_value = np.exp(inter_value)
            prob_results = inter_value / np.sum(inter_value)
            pred_index = np.argmax(prob_results)
            return prob_results, pred_index, inter_results

    def batch_train(self, batch_data, tags, learning_rate):
        """
        :param batch_data:
        :param tags: the right answers of the batch, tags should be digits
        """
        batch_size = len(tags)
        if batch_data.shape[0] != batch_size:
            print("Wrong training data!")
            return None
        total_loss = 0
        net_loss = 0

        # batch_data = batch_data[:8]
        # batch_size = len(batch_data)
        if self._is_properly_init:
            for idx, input in enumerate(batch_data):
                (total_loss_tmp, net_loss_tmp) = self._back_propagate(input, tags[idx], learning_rate)
                total_loss += total_loss_tmp
                net_loss += net_loss_tmp
            self._apply_propagation(batch_size, learning_rate)

        total_loss /= batch_size
        net_loss /= batch_size
        return total_loss, net_loss

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
    data = data.reshape(data.shape[0], 3072)
    data = np.array(data)/255
    data = data - 0.5
    return data

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def pca(data_train, data_test, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data_train)
    data_train_pca = pca.transform(data_train)
    data_test_pca = pca.transform(data_test)
    return data_train_pca, data_test_pca

def Hog(data_train, data_test):
    figure_data_train = data_train.reshape(len(data_train), 3, 32, 32).transpose(0, 2, 3, 1)
    data_train_hog = []
    for i in range(len(data_train)):
        data_train_hog.append(hog(figure_data_train[i], orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                               visualise=False))
    data_train_hog = np.array(data_train_hog)

    figure_data_test = data_test.reshape(len(data_test), 3, 32, 32).transpose(0, 2, 3, 1)
    data_test_hog = []
    for i in range(len(data_test)):
        data_test_hog.append(hog(figure_data_test[i], orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                              visualise=False))
    data_test_hog = np.array(data_test_hog)
    return data_train_hog, data_test_hog

# 获取镜像数据
def getMirror(data, label):
    figure_data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)
    data_new = []
    label_new = []
    for i in range(len(data)):
        image = figure_data[i]
        image_mirror = image[:, ::-1]
        image = image.reshape(3 * 32 * 32)
        image_mirror = image_mirror.reshape(3 * 32 * 32)
        data_new.append(image)
        data_new.append(image_mirror)
        label_new.append(label[i])
        label_new.append(label[i])
    data_new = np.array(data_new)
    label_new = np.array(label_new)
    return data_new, label_new

def LeakyRelu(x):
    for i in range(len(x)):
        if(x[i] < 0):
            x[i] *= 0.01
    return x

def LeakyRelu_derivative(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0.01
        else:
            x[i] = 1
    return x

def Elu(x):
    alpha = 1
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = alpha * (np.exp(x[i]) - 1)
    return x

def Elu_derivative(x):
    alpha = 1
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = alpha * np.exp(x[i])
        else:
            x[i] = 1
    return x

for i in range(1,6):
    if i == 1:
        x_traint,y_train = loadOne("../cifar-10-batches-py/data_batch_1")
    else:
        x_temp,y_temp = loadOne("../cifar-10-batches-py/data_batch_" + str(i))
        x_traint = np.concatenate((x_traint, x_temp), axis=0)
        y_train = np.concatenate((y_train, y_temp), axis=0)
        
x_testt,y_test = loadOne("../cifar-10-batches-py/test_batch")
# x_train, x_test = Hog(x_traint, x_testt)
# x_traint = x_traint.reshape(x_traint.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
# x_traint = x_traint.reshape(x_traint.shape[0], 3072)
# x_testt = x_testt.reshape(x_testt.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
# x_testt = x_testt.reshape(x_testt.shape[0], 3072)
# x_train, x_test = pca(x_traint,x_testt,1024)
# x_train = np.round(x_train)
# x_test = np.round(x_test)
# x_train_mirror, y_train_mirror = getMirror(x_train, y_train)
# x_train, x_test = Hog(x_traint, x_testt)
# x_train_pca, x_test_pca = pca(x_train,x_test,1024)
# x = np.insert(x,len(x[0]),-1,axis = 1)
x_train = normalization(x_traint)
x_test = normalization(x_testt)

#开始训练
norm_method = 0
norm_ratio = 0
if norm_method == 1:
    norm_ratio = 0.0002
elif norm_method == 2:
    norm_ratio = 0.0001

net_layer_shapes = (3072,32, 10)
batch_size = 32
learning_rate = 0.03
learning_rate_decay = 0.93
batch_size_decay = 1
epoch = 50
activation = "elu"
parameter_initializer = "Xavier"
optimizer = "BGD"               # GD, BGD, SGD, Momentum, AdaGrad, Adam

print("batch_size = %d, epoch = %d\nnorm_method = %d, norm_ratio = %.5f\nlearning_rate = %.4f, learning_decay = %.3f\nactivation = %s, para_initializer = %s, optimizer = %s\n" 
                    %(batch_size, epoch, norm_method, norm_ratio, learning_rate, learning_rate_decay, activation, parameter_initializer, optimizer))

clsfir = softmax_classifier(net_layer_shapes = net_layer_shapes,
                            norm_ratio = norm_ratio,
                            norm_method = norm_method, 
                            activation = activation, 
                            parameter_initializer = parameter_initializer,
                            optimizer = optimizer)
loss = []
acc = []

for i in range(1, epoch+1):
    time_start = time.time()
    n = 0
    loss_all = 0
    loss_net = 0
    while (n + batch_size < len(x_train)):
        x_temp = x_train[n: n + batch_size]
        y_temp = y_train[n: n + batch_size]

        loss_all_tmp, loss_net_tmp = clsfir.batch_train(x_temp, y_temp, learning_rate)
        loss_all += loss_all_tmp
        loss_net += loss_net_tmp

        n += batch_size
    loss_present = float(loss_all) / ((n / batch_size) + 1)
    loss_net_present = float(loss_net) / ((n / batch_size) + 1)
    loss.append(loss_present)
    result = []
    for j in range(10000):
        predict = clsfir.predict(x_test[j])[1]
        result.append(predict)

    num_test= 10000
    num_correct = np.sum(result == y_test) #计算准确率
    accuracy = float(num_correct) / num_test
    acc.append(accuracy)

    permutation = np.random.permutation(y_train.shape[0])
    x_train = x_train[permutation]
    y_train = y_train[permutation]
    
    learning_rate *= learning_rate_decay
    batch_size = int(batch_size*batch_size_decay)
    if batch_size == 0:
        batch_size = 1

    time_end = time.time()
    
    print("for epoch %d, loss is %.4f, net loss is %.4f, acc is %.3f, time cost %.2fs" %(i, loss_present, loss_net_present, accuracy, time_end-time_start))
    
#     plt.plot(loss)
#     plt.show()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(loss)
plt.show()
    
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(acc)
plt.show()

result = []
for i in range(50000):
    predict = clsfir.predict(x_train[i])[1]
    result.append(predict)
    
num_test= 50000
num_correct = np.sum(result == y_train) #计算准确率
accuracy = float(num_correct) / num_test
print("train acc: %f" % accuracy)


result = []
for i in range(10000):
    predict = clsfir.predict(x_test[i])[1]
    result.append(predict)
    
num_test= 10000
num_correct = np.sum(result == y_test) #计算准确率
accuracy = float(num_correct) / num_test
print("test acc: %f" % accuracy)

def counter(data):
    count = [0]*250
    for i in range(len(data)):
        t = data[i]
        t = int((data[i] + 1.25) * 100)
        count[t] += 1
    for i in range(len(count)):
        count[i] /= len(data)
    return count

def pltCounter(w):
    data = []
    for i in range(len(w)):
        data.append(w[i].flatten())
    data = np.array(data)
    for i in range(len(data)):
        c = counter(data[i])
        index = np.array(list(range(250))).astype('float64')
        index -= 125
        plt.subplot(len(data),1,i+1)
        plt.title("w" + str(i))
        plt.xlabel("w*100")
        plt.ylabel("probability")
        plt.bar(index,c)
        plt.show()

w = np.array(clsfir._net_weights)
pltCounter(w)

def plotMeanAndStd(w):
    data = []
    for i in range(len(w)):
        data.append(w[i].flatten())
    data = np.array(data)
    means = []
    stds = []
    for i in range(len(data)):
        means.append(np.mean(data[i]))
        stds.append(np.std(data[i]))
    index = range(1,len(data)+1)
    plt.subplot(1,1,1)
    plt.title("layer mean")
    plt.scatter(index, means) 
    plt.plot(index, means)
    plt.show()

    plt.subplot(2,1,2)
    plt.title("layer std")
    plt.scatter(index, stds) 
    plt.plot(index, stds)
    plt.show()

plotMeanAndStd(w)
