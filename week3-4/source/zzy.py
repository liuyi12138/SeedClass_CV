import numpy as np
# import cupy as np
import time
from data_process import loadOne, unpickle, normalization, LeakyRelu, Elu, LeakyRelu_derivative, Elu_derivative
from matplotlib import pyplot as plt


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

    def train(self, x_train, y_train, x_test, y_test, batch_size = 64, epoch = 50, learning_rate = 0.03, learning_rate_decay = 0.90, log_handler = None, cnt = None):
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

                loss_all_tmp, loss_net_tmp = self.batch_train(x_temp, y_temp, learning_rate)
                loss_all += loss_all_tmp
                loss_net += loss_net_tmp

                n += batch_size
            loss_present = float(loss_all) / ((n / batch_size) + 1)
            loss_net_present = float(loss_net) / ((n / batch_size) + 1)
            loss.append(loss_present)
            result = []
            for j in range(10000):
                predict = self.predict(x_test[j])[1]
                result.append(predict)

            num_test= 10000
            num_correct = np.sum(result == y_test) #计算准确率
            accuracy = float(num_correct) / num_test
            acc.append(accuracy)

            permutation = np.random.permutation(y_train.shape[0])
            x_train = x_train[permutation]
            y_train = y_train[permutation]
            learning_rate *= learning_rate_decay

            time_end = time.time()
            
            print("for epoch %d, loss is %.4f, net loss is %.4f, acc is %.4f, time cost %.2fs" %(i, loss_present, loss_net_present, accuracy, time_end-time_start))
            fp.write("for epoch %d, loss is %.4f, net loss is %.4f, acc is %.4f, time cost %.2fs\n" %(i, loss_present, loss_net_present, accuracy, time_end-time_start))
                
        plt.figure(figsize = (6, 4))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(range(1, epoch+1), loss)
        loss_figname = "../results/loss_test-latest" + str(cnt) + ".png"
        plt.savefig(loss_figname)
        plt.close()

        plt.figure(figsize = (6, 4))
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.plot(range(1, epoch+1), acc)
        acc_figname = "../results/acc_test-latest" + str(cnt) + ".png"
        plt.savefig(acc_figname)
        plt.close()



#数据获取
for i in range(1,6):
    if i == 1:
        x_traint,y_train = loadOne("../../../cifar-10-batches-py/data_batch_1")
    else:
        x_temp,y_temp = loadOne("../../../cifar-10-batches-py/data_batch_" + str(i))
        x_traint = np.concatenate((x_traint, x_temp), axis=0)
        y_train = np.concatenate((y_train, y_temp), axis=0)
        
x_testt,y_test = loadOne("../../../cifar-10-batches-py/test_batch")
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

shape_list = [(3072,32,10), (3072,64,10)]
lr_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
bat_list = [32,64]
# init_list = ["Xavier"]
learning_decay_List = [0.933, 0.89, 1]

cnt = 0
for net_layer_shapes in shape_list:
    for lr in lr_list:
        for batch_size in bat_list:
            if net_layer_shapes == (3072,32,10) and batch_size == 32:
                continue
            if net_layer_shapes == (3072,64,10) and batch_size == 32:
                continue
            for learning_rate_decay in learning_decay_List:
                cnt += 1
                learning_rate = lr
                print("The %d test start:\n" %cnt)
                #开始训练
                norm_method = 0
                norm_ratio = 0
                if norm_method == 1:
                    norm_ratio = 0.0002
                elif norm_method == 2:
                    norm_ratio = 0.0001

                # net_layer_shapes = (3072, 32, 10)
                # batch_size = 16
                # learning_rate = 0.05
                # learning_rate_decay = 0.9
                epoch = 20
                activation = "relu"
                parameter_initializer = "Xavier"
                optimizer = "BGD"               # GD, BGD, SGD, Momentum, AdaGrad, Adam

                print("net_shape = ", net_layer_shapes, "batch_size = %d, epoch = %d\nnorm_method = %d, norm_ratio = %.5f\nlearning_rate = %.4f, learning_decay = %.3f\nactivation = %s, para_initializer = %s, optimizer = %s\n" 
                                    %(batch_size, epoch, norm_method, norm_ratio, learning_rate, learning_rate_decay, activation, parameter_initializer, optimizer))

                fp = open("../results/log-latest.txt", "a+")
                fp.write("The %d test\n" %cnt)
                fp.write("net_shape = (%d %d %d)" %(net_layer_shapes[0], net_layer_shapes[1], net_layer_shapes[2]))
                fp.write(", batch_size = %d, epoch = %d\nnorm_method = %d, norm_ratio = %.5f\nlearning_rate = %.3f, learning_decay = %.3f\nactivation = %s, para_initializer = %s, optimizer = %s\n" 
                                    %(batch_size, epoch, norm_method, norm_ratio, learning_rate, learning_rate_decay, activation, parameter_initializer, optimizer))

                clsfir = softmax_classifier(net_layer_shapes = net_layer_shapes,
                                            norm_ratio = norm_ratio,
                                            norm_method = norm_method, 
                                            activation = activation, 
                                            parameter_initializer = parameter_initializer,
                                            optimizer = optimizer)

                clsfir.train(x_train = x_train, y_train = y_train,
                            x_test = x_test, y_test = y_test,
                            batch_size = batch_size, 
                            epoch = epoch, 
                            learning_rate = learning_rate, 
                            learning_rate_decay = learning_rate_decay,
                            log_handler = fp,
                            cnt = cnt)

                #训练集
                result = []
                for i in range(50000):
                    predict = clsfir.predict(x_train[i])[1]
                    result.append(predict)
                    
                num_test= 50000
                num_correct = np.sum(result == y_train) #计算准确率
                train_accuracy = float(num_correct) / num_test
                print("train acc: %f" % train_accuracy)

                #测试集
                result = []
                for i in range(10000):
                    predict = clsfir.predict(x_test[i])[1]
                    result.append(predict)
                    
                num_test= 10000
                num_correct = np.sum(result == y_test) #计算准确率
                test_accuracy = float(num_correct) / num_test
                print("test acc: %f" % test_accuracy)
                fp.write("train acc: %f, test acc: %f\n\n" %(train_accuracy, test_accuracy))
                fp.close()