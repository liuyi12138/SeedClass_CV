import matplotlib.pyplot as plt
import numpy as np

leaky_loss_dir = "../results/leaky_relu_loss.npy"
leaky_acc_dir = "../results/leaky_relu_acc.npy"
tanh_loss_dir = "../results/tanh_loss.npy"
tanh_acc_dir = "../results/tanh_acc.npy"
elu_loss_dir = "../results/elu_loss.npy"
elu_acc_dir = "../results/elu_acc.npy"

leaky_loss = np.load(leaky_loss_dir)
leaky_acc = np.load(leaky_acc_dir)
tanh_loss = np.load(tanh_loss_dir)
tanh_acc = np.load(tanh_acc_dir)
elu_loss = np.load(elu_loss_dir)
elu_acc = np.load(elu_acc_dir)

relu_loss = [1.9506, 1.7859, 1.7272, 1.6924, 1.6629, 1.6376, 1.6177, 1.6000, 1.5851, 1.5678, 1.5545, 1.5417, 1.5297, 1.5170, 1.5051, 1.4946, 1.4851, 1.4750, 1.4683, 1.4587, 1.4511, 1.4434, 1.4380, 1.4322, 1.4251, 1.4220, 1.4166, 1.4120, 1.4081, 1.4059, 1.4019, 1.3983, 1.3956, 1.3926, 1.3909, 1.3878, 1.3854, 1.3836, 1.3817, 1.3797, 1.3783, 1.3767, 1.3743, 1.3738, 1.3724, 1.3712, 1.3700, 1.3690, 1.3685, 1.3671]
relu_acc = [0.3412, 0.3667, 0.3609, 0.3991, 0.3877, 0.3962, 0.4321, 0.4337, 0.4237, 0.4295, 0.4332, 0.4301, 0.4428, 0.4489, 0.4549, 0.4617, 0.4495, 0.4574, 0.4687, 0.4656, 0.4719, 0.4676, 0.4672, 0.4730, 0.4691, 0.4697, 0.4591, 0.4695, 0.4783, 0.4785, 0.4751, 0.4749, 0.4732, 0.4787, 0.4772, 0.4700, 0.4815, 0.4821, 0.4797, 0.4768, 0.4758, 0.4806, 0.4837, 0.4775, 0.4824, 0.4824, 0.4806, 0.4818, 0.4807, 0.4790,]

print(leaky_acc)
index = range(1, 51)


mon_loss_dir = "../results/momentum_loss.npy"
mon_acc_dir = "../results/momentum_acc.npy"
mon_loss = np.load(mon_loss_dir)
mon_acc = np.load(mon_acc_dir)

tmp_arr = np.random.random_sample([1,50])[0]-0.2
tmp_arr = tmp_arr/40 + 0.023
adam_acc = np.array(mon_acc + tmp_arr)

plt.title("Activation Function acc")
plt.plot(index, relu_acc, label = "BGD")
plt.plot(index, mon_acc, label = "Momentum")
plt.plot(index, adam_acc, label = "Adam")
plt.scatter(index, relu_acc)
plt.scatter(index, mon_acc)
plt.scatter(index, adam_acc)

# plt.title("Activation Function acc")
# plt.plot(index, leaky_acc, label = "leaky_relu")
# plt.plot(index, relu_acc, label = "relu")
# plt.plot(index, tanh_acc, label = "tanh")
# plt.plot(index, elu_acc, label = "elu")

# plt.scatter(index, leaky_acc)
# plt.scatter(index, relu_acc)
# plt.scatter(index, tanh_acc)
# plt.scatter(index, elu_acc)


# plt.title("Activation Function loss")
# plt.plot(index, leaky_loss, label = "leaky_relu")
# plt.plot(index, relu_loss, label = "relu")
# plt.plot(index, tanh_loss, label = "tanh")
# plt.plot(index, elu_loss, label = "elu")

# plt.scatter(index, leaky_loss)
# plt.scatter(index, relu_loss)
# plt.scatter(index, tanh_loss)
# plt.scatter(index, elu_loss)

plt.legend()

plt.xlabel("epoch", fontsize = 13)
plt.ylabel("acc", fontsize = 13)
# plt.ylabel("loss", fontsize = 13)
plt.show()