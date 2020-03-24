import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

class softmax_classifier(object):
    def __init__(self, net_layer_shapes, norm_ratio, norm_method):
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
            self._activation_method = "relu"

            weights_num = len(net_layer_shapes) - 1
            for i in range(1, weights_num + 1):
                weight_range = np.sqrt(6 / (net_layer_shapes[i - 1] + net_layer_shapes[i]))
                self._net_weights.append(np.random.uniform(-weight_range, weight_range, (net_layer_shapes[i - 1] + 1,
                                                                                         net_layer_shapes[
                                                                                             i])))  # Xavier initialization reference https://zhuanlan.zhihu.com/p/76602243
                self._pending_weights.append(np.zeros((net_layer_shapes[i - 1] + 1, net_layer_shapes[i])))

        # set up the partial derivatives

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

                # normalization
                norm_derivative_method = None
                if self._norm_method == 1:
                    norm_derivative_method = lambda x: np.sum(np.abs(x))
                elif self._norm_method == 2:
                    norm_derivative_method = lambda x: np.sum(x ** 2)

                if norm_derivative_method:
                    for weights in self._net_weights:
                        loss += self._norm_ratio * norm_derivative_method(weights)

                # activation function setup
                act_derivative = lambda y: 1
                if self._activation_method == "relu":
                    act_derivative = lambda y: (y > 0)
                elif self._activation_method == "tanh":
                    act_derivative = lambda y: 1 - np.multiply(y, y)

                # w += (x.T).dot(p-one_hot)
                current_derivative = np.mat(prob_results - one_hot_tag)
                for i in range(-1, -len(self._net_weights), -1):  # totally len(self._net_weights)-1
                    self._pending_weights[i] -= np.mat(np.concatenate((inter_results[i - 1], [1]))).T.dot(
                        current_derivative
                    ) * learning_rate
                    # derivative of results of activation function, so the biases is ignored here
                    current_derivative = current_derivative.dot(
                        self._net_weights[i][:-1, :].T)
                    # derivative of activation function, given function results
                    current_derivative = np.multiply(act_derivative(np.array(inter_results[i - 1])), current_derivative)

                self._pending_weights[0] -= np.mat(np.concatenate((input, [1]))).T.dot(
                    current_derivative) * learning_rate

                # next
                return loss, net_loss

    def _apply_propagation(self, division, learning_rate):
        """
        apply the temporary parameters updates to the weights and biases
        :param division
        """
        # normal regularization
        norm_func = None
        if self._norm_method == 1:
            norm_func = lambda x: self._norm_ratio * np.sign(x) * learning_rate
        elif self._norm_method == 2:
            norm_func = lambda x: self._norm_ratio * 2 * x * learning_rate

        if self._is_properly_init:
            # update all the propagation
            for i in range(len(self._net_weights)):
                if norm_func:
                    self._net_weights[i] -= norm_func(self._net_weights[i])

                self._net_weights[i] += self._pending_weights[i] / division
                self._pending_weights[i] *= 0  # reset the weights to zeros

    def predict(self, input, is_return_inter_values=False):
        """

        :param input: data input
        :param is_return_inter_values: when True, this function returns the intermediate values for back propagation
        :return inter_results stores the intermediate values of those layers except input
        """
        if self._is_properly_init:
            inter_value = input
            inter_results = []

            # activation function setup
            act_func = lambda x: x
            if self._activation_method == "relu":
                act_func = lambda x: np.multiply(x, (x > 0))
            if self._activation_method == "tanh":
                act_func = lambda x: np.tanh(x)

            for idx, weight_mat in enumerate(self._net_weights):
                inter_value = np.concatenate((inter_value, [1])).dot(weight_mat)

                if idx != len(self._net_weights) - 1:
                    inter_value = act_func(inter_value)

                if is_return_inter_values: inter_results.append(
                    inter_value)  # the inter_values are results of activation function

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

        if self._is_properly_init:
            for idx, input in enumerate(batch_data):
                (total_loss_tmp, net_loss_tmp) = self._back_propagate(input, tags[idx], learning_rate)
                total_loss += total_loss_tmp
                net_loss += net_loss_tmp
            self._apply_propagation(batch_size, learning_rate)

        total_loss /= batch_size
        net_loss /= batch_size
        return total_loss, net_loss

def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def classify(X, y):
    clf = linear_model.LinearRegression()
    clf.fit(X, y)
    return clf

def visualize(X, y, clf):
#     plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
#     plt.show()
    plot_decision_boundary(lambda x: clf.predict(x), X, y)
#     plt.title("Logistic Regression")

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

def my_visualize(X, y, clf):
#     plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
#     plt.show()
    my_plot_decision_boundary(lambda x: clf.predict(x), X, y)
#     plt.title("Logistic Regression")
    
def my_plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = []
    allPoint = np.c_[xx.ravel(), yy.ravel()]
    for i in range(len(allPoint)):
        Z.append(pred_func(allPoint[i])[1])
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

# 数据生成
X, y = generate_data()

norm_method = 0
norm_ratio = 0
if norm_method == 1:
    norm_ratio = 0.0002
elif norm_method == 2:
    norm_ratio = 0.0001

batch_size = 10
learning_rate = 0.01
epoch = 1000

clsfir = softmax_classifier((2,128,64,16,8,2),norm_ratio,norm_method)
loss = []

plt.xlabel('Epoch')
plt.ylabel('Loss')

for i in range(epoch):
    n = 0
    loss_all = 0
    loss_net = 0
    while (n + batch_size < len(X)):
        x_temp = X[n: n + batch_size]
        y_temp = y[n: n + batch_size]

        loss_all_tmp, loss_net_tmp = clsfir.batch_train(x_temp, y_temp, learning_rate)
        loss_all += loss_all_tmp
        loss_net += loss_net_tmp

        n += batch_size
    loss_present = float(loss_all) / ((n / batch_size) + 1)
    loss_net_present = float(loss_net) / ((n / batch_size) + 1)

    loss.append(loss_present)

    permutation = np.random.permutation(y.shape[0])
    X = X[permutation]
    y = y[permutation]
    
plt.plot(loss)
plt.show()

# 示例
clf = classify(X, y)
visualize(X, y, clf)
# 我们的代码
my_visualize(X, y, clsfir)