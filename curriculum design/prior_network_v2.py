import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import time

def conv2d(inputs, num_outputs, kernel_shape, strides=[1, 1], mask_type=None, scope="conv2d"):
  """
  Args:
    inputs: nhwc
    kernel_shape: [height, width]
    mask_type: None or 'A' or 'B' or 'C'
  Returns:
    outputs: nhwc
  """
  with tf.compat.v1.variable_scope(scope) as scope:
    kernel_h, kernel_w = kernel_shape
    stride_h, stride_w = strides
    batch_size, height, width, in_channel = inputs.get_shape().as_list()

    center_h = kernel_h // 2
    center_w = kernel_w // 2

    assert kernel_h % 2 == 1 and kernel_w % 2 == 1, "kernel height and width must be odd number"
    mask = np.zeros((kernel_h, kernel_w, in_channel, num_outputs), dtype=np.float32)
    if mask_type is not None:
      #C
      mask[:center_h, :, :, :] = 1
      if mask_type == 'A':
        mask[center_h, :center_w, :, :] = 1
        """
        mask[center_h, :center_w, :, :] = 1
        #G channel
        mask[center_h, center_w, 0:in_channel:3, 1:num_outputs:3] = 1
        #B Channel
        mask[center_h, center_w, 0:in_channel:3, 2:num_outputs:3] = 1
        mask[center_h, center_w, 1:in_channel:3, 2:num_outputs:3] = 1
        """
      if mask_type == 'B':
        mask[center_h, :center_w+1, :, :] = 1
        """
        mask[center_h, :center_w, :, :] = 1
        #R Channel
        mask[center_h, center_w, 0:in_channel:3, 0:num_outputs:3] = 1
        #G channel
        mask[center_h, center_w, 0:in_channel:3, 1:num_outputs:3] = 1
        mask[center_h, center_w, 1:in_channel:3, 1:num_outputs:3] = 1
        #B Channel
        mask[center_h, center_w, :, 2:num_outputs:3] = 1
        """
    else:
      mask[:, :, :, :] = 1

    weights_shape = [kernel_h, kernel_w, in_channel, num_outputs]
    # create weight, guass distriution
    weights = tf.compat.v1.get_variable("weights", weights_shape,
      tf.float32, tf.compat.v1.truncated_normal_initializer(stddev=0.1))
    weights = weights * mask
    # gen tensor constant
    biases = tf.compat.v1.get_variable("biases", [num_outputs],
          tf.float32, tf.compat.v1.constant_initializer(0.0))

    outputs = tf.nn.conv2d(input=inputs, filters=weights, strides=[1, stride_h, stride_w, 1], padding="SAME")
    outputs = tf.nn.bias_add(outputs, biases)

    return outputs

def gated_conv2d(inputs, state, kernel_shape, scope):
  """
  Args:
    inputs: nhwc
    state:  nhwc
    kernel_shape: [height, width]
  Returns:
    outputs: nhwc
    new_state: nhwc
  """
  with tf.compat.v1.variable_scope(scope) as scope:
    batch_size, height, width, in_channel = inputs.get_shape().as_list()
    kernel_h, kernel_w = kernel_shape
    #state route
    left = conv2d(state, 2 * in_channel, kernel_shape, strides=[1, 1], mask_type='C', scope="conv_s1")
    left1 = left[:, :, :, 0:in_channel]
    left2 = left[:, :, :, in_channel:]
    left1 = tf.nn.tanh(left1)
    left2 = tf.nn.sigmoid(left2)
    new_state = left1 * left2
    left2right = conv2d(left, 2 * in_channel, [1, 1], strides=[1, 1], scope="conv_s2")
    #input route
    right = conv2d(inputs, 2 * in_channel, [1, kernel_w], strides=[1, 1], mask_type='B', scope="conv_r1")
    right = right + left2right
    right1 = right[:, :, :, 0:in_channel]
    right2 = right[:, :, :, in_channel:]
    right1 = tf.nn.tanh(right1)
    right2 = tf.nn.sigmoid(right2)
    up_right = right1 * right2
    up_right = conv2d(up_right, in_channel, [1, 1], strides=[1, 1], mask_type='B', scope="conv_r2")
    outputs = inputs + up_right

    return outputs, new_state

def batch_norm(x, train=True, scope=None):
  return tf.contrib.layers.batch_norm(x, center=True, scale=True, updates_collections=None, is_training=train, trainable=True, scope=scope)

def resnet_block(inputs, num_outputs, kernel_shape, strides=[1, 1], scope=None, train=True):
  """
  Args:
    inputs: nhwc
    num_outputs: int
    kernel_shape: [kernel_h, kernel_w]
  Returns:
    outputs: nhw(num_outputs)
  """
  with tf.compat.v1.variable_scope(scope) as scope:
    conv1 = conv2d(inputs, num_outputs, kernel_shape, strides=[1, 1], mask_type=None, scope="conv1")
    bn1 = batch_norm(conv1, train=train, scope='bn1')
    relu1 = tf.nn.relu(bn1)
    conv2 = conv2d(relu1, num_outputs, kernel_shape, strides=[1, 1], mask_type=None, scope="conv2")
    bn2 = batch_norm(conv2, train=train, scope='bn2')
    output = inputs + bn2

    return output 

def deconv2d(inputs, num_outputs, kernel_shape, strides=[1, 1], scope="deconv2d"):
  """
  Args:
    inputs: nhwc
    num_outputs: int
    kernel_shape: [kernel_h, kernel_w]
    strides: [stride_h, stride_w]
  Returns:
    outputs: nhwc
  """
  with tf.compat.v1.variable_scope(scope) as scope:
    return tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_shape, strides, \
          padding='SAME', weights_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1), \
          biases_initializer=tf.compat.v1.constant_initializer(0.0))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1) # only difference

def logits_2_pixel_value(logits, mu=1.1):
  """
  Args:
    logits: [n, 256] float32
    mu    : float32
  Returns:
    pixels: [n] float32
  """
  rebalance_logits = logits * mu
  probs = softmax(rebalance_logits)
  pixel_dict = np.arange(0, 256, dtype=np.float32)
  pixels = np.sum(probs * pixel_dict, axis=1)
  return np.floor(pixels)

class Net(object):
  def __init__(self, hr_images, lr_images, scope):
    """
    Args:[0, 255]
      hr_images: [batch_size, hr_height, hr_width, in_channels] float32
      lr_images: [batch_size, lr_height, lr_width, in_channels] float32
    """
    with tf.compat.v1.variable_scope(scope) as scope:
      self.train = tf.compat.v1.placeholder(tf.bool)
      hr_images = tf.cast(hr_images, tf.float32)
      lr_images = tf.cast(lr_images, tf.float32)
      self.construct_net(hr_images, lr_images)

  def prior_network(self, hr_images):
    """
    Args:[-0.5, 0.5]
      hr_images: [batch_size, hr_height, hr_width, in_channels]
    Returns:
      prior_logits: [batch_size, hr_height, hr_width, 3*256]
    """
    with tf.compat.v1.variable_scope('prior') as scope:
      conv1 = conv2d(hr_images, 64, [7, 7], strides=[1, 1], mask_type='A', scope="conv1")
      inputs = conv1
      state = conv1
      for i in range(20):
        inputs, state = gated_conv2d(inputs, state, [5, 5], scope='gated' + str(i))
      conv2 = conv2d(inputs, 1024, [1, 1], strides=[1, 1], mask_type='B', scope="conv2")
      conv2 = tf.nn.relu(conv2)
      prior_logits = conv2d(conv2, 3 * 256, [1, 1], strides=[1, 1], mask_type='B', scope="conv3")

      prior_logits = tf.concat([prior_logits[:, :, :, 0::3], prior_logits[:, :, :, 1::3], prior_logits[:, :, :, 2::3]], 3)

      return prior_logits

  def conditioning_network(self, lr_images):
    """
    Args:[-0.5, 0.5]
      lr_images: [batch_size, lr_height, lr_width, in_channels]
    Returns:
      conditioning_logits: [batch_size, hr_height, hr_width, 3*256]
    """
    res_num = 6
    with tf.compat.v1.variable_scope('conditioning') as scope:
      inputs = lr_images
      inputs = conv2d(inputs, 32, [1, 1], strides=[1, 1], mask_type=None, scope="conv_init")
      for i in range(2):
        for j in range(res_num):
          inputs = resnet_block(inputs, 32, [3, 3], strides=[1, 1], scope='res' + str(i) + str(j), train=self.train)
        inputs = deconv2d(inputs, 32, [3, 3], strides=[2, 2], scope="deconv" + str(i))
        inputs = tf.nn.relu(inputs)
      for i in range(res_num):
        inputs = resnet_block(inputs, 32, [3, 3], strides=[1, 1], scope='res3' + str(i), train=self.train)
      conditioning_logits = conv2d(inputs, 256, [1, 1], strides=[1, 1], mask_type=None, scope="conv")

      return conditioning_logits

  def softmax_loss(self, logits, labels):
    logits = tf.reshape(logits, [-1, 256])
    labels = tf.cast(labels, tf.int32)
    labels = tf.reshape(labels, [-1])
    return tf.compat.v1.losses.sparse_softmax_cross_entropy(
           labels, logits)
           
  def construct_net(self, hr_images, lr_images):
    """
    Args: [0, 255]
    """
    #labels
    labels = hr_images
    #normalization images [-0.5, 0.5]
    hr_images = hr_images / 255.0 - 0.5
    lr_images = lr_images / 255.0 - 0.5
    # self.conditioning_logits = self.conditioning_network(lr_images)
    self.prior_logits = self.prior_network(hr_images)

    self.loss = self.softmax_loss(self.prior_logits, labels)

    tf.compat.v1.summary.scalar('loss', self.loss)

def show_samples(np_imgs):
  """
  Args:
    np_imgs: [N, H, W, 3] float32
    img_path: str
  """
  np_imgs = np_imgs.astype(np.uint8)
  N, H, W, _ = np_imgs.shape
  num = int(N ** (0.5))
  merge_img = np.zeros((num * H, num * W, 3), dtype=np.uint8)
  for i in range(num):
    for j in range(num):
      merge_img[i*H:(i+1)*H, j*W:(j+1)*W, :] = np_imgs[i*num+j,:,:,:]

  plt.imshow(merge_img,plt.cm.gray)

# data process
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()
hr_images = []
for i in range(len(x_train)):
    if y_train[i] == 3:
        hr_images.append(x_train[i])
hr_images = np.array(hr_images)

lr_images = []
for i in range(len(hr_images)):
    lr_images.append(transform.resize(hr_images[i], (7, 7)))
lr_images = np.array(lr_images)

hr_images = np.resize(hr_images,(hr_images.shape[0],28,28,1))
lr_images = np.resize(lr_images,(lr_images.shape[0],7,7,1))
hr_images = hr_images.astype(float)
lr_images = lr_images.astype(float)

# model construction
xs = tf.compat.v1.placeholder(float, [None, 28, 28, 1])
ys = tf.compat.v1.placeholder(float, [None, 7, 7, 1])
net = Net(xs, ys, 'prsr')

# parameters
epoch = 300
learning_rate = 4e-4
batch_size = 32
global_step = tf.compat.v1.get_variable('global_step', [], initializer=tf.compat.v1.constant_initializer(0), trainable=False)
learning_rate_decay = tf.compat.v1.train.exponential_decay(learning_rate, global_step,
                                   500000, 0.5, staircase=True)
optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate_decay, decay=0.95, momentum=0.9, epsilon=1e-8)
train_op = optimizer.minimize(net.loss, global_step=global_step)

init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
summary_op = tf.compat.v1.summary.merge_all()
saver = tf.compat.v1.train.Saver()
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.run(init_op)
coord = tf.train.Coordinator()
threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

# train
Loss = []
with tf.device('/gpu:0'):
    epochIters = 0
    mu=1.1
    while not coord.should_stop() and epochIters <= epoch:
        # Run training steps or whatever
        batchIters = 1
        while batchIters*batch_size <= len(hr_images):
            temp_hr_images = hr_images[(batchIters-1)*batch_size:batchIters*batch_size]
            temp_lr_images = lr_images[(batchIters-1)*batch_size:batchIters*batch_size]
            t1 = time.time()
            _, loss = sess.run([train_op, net.loss], feed_dict={xs: temp_hr_images, ys: temp_lr_images, net.train: True})
            t2 = time.time()
            if(batchIters % 30 == 0):
                print('epoch %d batch %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' % ((epochIters, batchIters, loss, batch_size/(t2-t1), (t2-t1))))
                Loss.append(loss)
            batchIters += 1
            
        epochIters += 1
        permutation = np.random.permutation(hr_images.shape[0])
        hr_images = hr_images[permutation, :, :, :]
        lr_images = lr_images[permutation, :, :, :]

        if epochIters > epoch:
            coord.request_stop()


plt.plot(Loss)
plt.show()

# build test
test_hr_images = []
for i in range(len(x_test)):
    if y_test[i] == 3:
        test_hr_images.append(x_test[i])
test_hr_images = np.array(test_hr_images)

test_lr_images = []
for i in range(len(test_hr_images)):
    test_lr_images.append(transform.resize(test_hr_images[i], (7, 7)))
test_lr_images = np.array(test_lr_images)

test_hr_images = np.resize(test_hr_images,(test_hr_images.shape[0],28,28,1))
test_lr_images = np.resize(test_lr_images,(test_lr_images.shape[0],7,7,1))
test_hr_images = test_hr_images.astype(float)
test_lr_images = test_lr_images.astype(float)

# predict test
c_logits = net.conditioning_logits
lr_imgs = test_lr_images[0:32]
hr_imgs = test_hr_images[0:32]
#np_hr_imgs, np_lr_imgs = sess.run([hr_imgs_tf, lr_imgs_tf])
gen_hr_imgs = np.zeros((batch_size, 28, 28, 1), dtype=np.float32)
#gen_hr_imgs = np_hr_imgs
#gen_hr_imgs[:,16:,16:,:] = 0.0
np_c_logits = sess.run(c_logits, feed_dict={xs: hr_imgs, ys: lr_imgs, net.train:False})
for i in range(28):
  for j in range(28):
    for c in range(1):
      new_pixel = logits_2_pixel_value(np_c_logits[:, i, j, c*256:(c+1)*256], mu=mu)
      gen_hr_imgs[:, i, j, c] = new_pixel

# output all
plt.subplot(1, 3, 1)
show_samples(hr_imgs)
plt.subplot(1, 3, 2)
show_samples(lr_imgs)
plt.subplot(1, 3, 3)
show_samples(gen_hr_imgs)

# output single
index = 7
hr_imgg = gen_hr_imgs[index]
lr_imgt = test_lr_images[index]
hr_imgt = test_hr_images[index]
hr_imgg = np.resize(hr_imgg,(28,28))
hr_imgt = np.resize(hr_imgt,(28,28))
lr_imgt = np.resize(lr_imgt,(7,7))

plt.subplot(1, 3, 1)
plt.imshow(hr_imgt,plt.cm.gray)
plt.subplot(1, 3, 2)
plt.imshow(lr_imgt,plt.cm.gray)
plt.subplot(1, 3, 3)
plt.imshow(hr_imgg,plt.cm.gray)