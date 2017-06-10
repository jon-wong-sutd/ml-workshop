# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Create the model
x = tf.placeholder(tf.float32, [None, 784])

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  import numpy as np

  # Take 10 datapoints for each number.
  numbers = [[] for i in range(10)]
  for i in range(len(mnist.train.labels)):
    if sum(len(x) for x in numbers) == 100:
      break

    number = np.where(mnist.train.labels[i] == 1)[0][0]
    if len(numbers[number]) < 10:
      numbers[number].append(i)

  import vis_mnist as vm
  # for i in range(10):
  #   vm.show_image(numbers[i])

  # Scramble subset.
  numbers = np.asarray(numbers)
  numbers = numbers.reshape(100)
  np.random.shuffle(numbers)

  subset_images = []
  subset_labels = []
  for i in numbers:
    subset_images.append(mnist.train.images[i])
    subset_labels.append(mnist.train.labels[i])

  mnist.train._images = np.asarray(subset_images)
  mnist.train._labels = np.asarray(subset_labels)
  mnist.train._num_examples = 100

  # Train
  for _ in range(400):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob:0.5})

  # Test trained model
  print('Initial accuracy: {}'.format(sess.run(accuracy,
                                        feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels, keep_prob: 1.0})))

  for _ in range(400):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob:0.5})

  # Test trained model
  print('2nd accuracy: {}'.format(sess.run(accuracy,
                                        feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels, keep_prob: 1.0})))

  for i in range(2):
    if FLAGS.data_aug is True:
      import elastic_deform as ed
      # Deform all images first.
      print("Deforming all 'train' images..")
      for i in range(len(mnist.train.images)):
        new_image = ed.deform(mnist.train.images[i].reshape((28, 28)))
        mnist.train.images[i] = new_image.reshape(784)
        print('Processed image {}'.format(i), end='\r')
      print("\nDeformation done.")

    for _ in range(400):
      batch_xs, batch_ys = mnist.train.next_batch(50)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob:0.5})

    # Test trained model
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  FLAGS.data_aug = True
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

  # With data augmentation...
  #   0.8002, 0.8077, 0.8267
  #   0.7842, 0.7903, 0.8041
  #   0.7887, 0.7914, 0.8070
  #   0.8037, 0.8179, 0.8178, 0.8199
  #   0.7958, 0.8042, 0.8012, 0.7969
  #   0.7994, 0.8058, 0.8247, 0.8286
  # Without...
  #   0.7712, 0.7814, 0.7907
  #   0.7726, 0.7902, 0.7974
  #   0.7910, 0.7989, 0.8043
  # No difference.
