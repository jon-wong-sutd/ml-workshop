import tensorflow as tf

# 4 neurons for 4 features: diamond, hor, ver, cross.
num_conv1 = 4
conv1_kernel_size = 7
# 2 neurons in layer 2. Only 2 classes to recognize.
num_conv2 = 2
conv2_kernel_size = 2
# 10 classes to classify into.
num_classes = 10

stride = 7

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Create the model
x = tf.placeholder(tf.float32, [None, 784])

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

W_conv1 = weight_variable([conv1_kernel_size, conv1_kernel_size, 1, num_conv1])
b_conv1 = bias_variable([num_conv1])
# Activation map 4 x 4

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, conv1_kernel_size) + b_conv1)

W_conv2 = weight_variable([conv2_kernel_size, conv2_kernel_size, num_conv1, num_conv2])
b_conv2 = bias_variable([num_conv2])
# Activation map 2 x 2
final_activation_size = 2 * 2

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, conv2_kernel_size) + b_conv2)

# fc_range = 1024
fc_range = 5
W_fc1 = weight_variable([final_activation_size * num_conv2, fc_range])
b_fc1 = bias_variable([fc_range])

h_pool2_flat = tf.reshape(h_conv2, [-1, final_activation_size * num_conv2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([fc_range, num_classes])
b_fc2 = bias_variable([num_classes])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
