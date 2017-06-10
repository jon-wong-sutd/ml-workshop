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
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

import tensorflow as tf

import elastic_deform as ed
import mnist_data as md
import numpy as np

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
  acc_summ = tf.summary.scalar('accuracy', accuracy)

  # Accuracy trackers
  acc1 = tf.get_variable('accuracy1', [], dtype=tf.float32)
  acc2 = tf.get_variable('accuracy2', [], dtype=tf.float32)
  acc3 = tf.get_variable('accuracy3', [], dtype=tf.float32)
  acc4 = tf.get_variable('accuracy4', [], dtype=tf.float32)
  improvement = tf.divide(tf.subtract(acc4, acc3), tf.subtract(acc2, acc1))
  impr_summ = tf.summary.scalar('improvement', improvement)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Store all variables' values now. We want each run to be the same, not random.
  saver = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2])
  init_save = 'saves/init_values'
  saver.save(sess, init_save)

  writer = tf.summary.FileWriter('./log')
  normal_writer = tf.summary.FileWriter('./log/normal')
  expanded_writer = tf.summary.FileWriter('./log/expanded')

  def train(train_steps, batch_size, normal_dataset, expanded_dataset, step=0):
    # Initialize all weights first.
    saver.restore(sess, init_save)

    for _ in range(train_steps):
      batch_xs, batch_ys = expanded_dataset.next_batch(batch_size, shuffle=False)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob:0.5})

    # Test trained model
    summ, acc_val = sess.run([acc_summ, tf.assign(acc3, accuracy)],
               feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    print('Data Aug accuracy: {}'.format(acc_val))

    for _ in range(train_steps):
      batch_xs, batch_ys = expanded_dataset.next_batch(batch_size, shuffle=False)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob:0.5})

    # Test trained model
    summ, acc_val = sess.run([acc_summ, tf.assign(acc4, accuracy)],
               feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    print('Data Aug accuracy: {}'.format(acc_val))
    expanded_writer.add_summary(summ, step)

    saver.restore(sess, init_save) # Reset for run with larger dataset

    # Train
    for _ in range(train_steps):
      batch_xs, batch_ys = normal_dataset.next_batch(batch_size, shuffle=False)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob:0.5})

    # Test trained model
    summ, acc_val = sess.run([acc_summ, tf.assign(acc1, accuracy)],
                feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    print('Non-aug accuracy: {}'.format(acc_val))

    for _ in range(train_steps):
      batch_xs, batch_ys = normal_dataset.next_batch(batch_size, shuffle=False)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob:0.5})

    # Test trained model
    summ, acc_val = sess.run([acc_summ, tf.assign(acc2, accuracy)],
                feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    print('Non-aug accuracy: {}'.format(acc_val))
    normal_writer.add_summary(summ, step)

    summ, impr = sess.run([impr_summ, improvement])
    return summ, impr

  for i in range(10):
    # 50 for train_steps seems to hit plateau. Overfitting for dataset 10 (for each digit).
    # summ, impr = train((i + 1) * 10)

    # Now, we vary dataset size, and train_steps in sync.
    # 5 times over entire epochs isn't working too well. Try 20.
    # Ensure each train step uses an entire epoch (normal dataset) 20 times.
    n = 10 * (i + 1)
    # Select data once. Want to check whether results can be reproduced exactly.
    normal_dataset, expanded_dataset = md.select_data(n, True)

    batch_size = 50
    train_steps = ((n * 10) // batch_size + 1) * 20
    print('train_steps: {}'.format(train_steps))
    summ, impr = train(train_steps, batch_size, normal_dataset, expanded_dataset, i)
    writer.add_summary(summ, i)
    print('Impr {}: {}'.format(i, impr))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='MNIST-data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  FLAGS.data_aug = True
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
