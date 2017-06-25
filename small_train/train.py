from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

# from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

import tensorflow as tf

import graph as g
import trainset_2elems as trainset
import utils

import numpy as np
import math
from pathlib import Path

import matplotlib.pyplot as plt

# import sys
# sys.path.insert(0, '../')
# import mnist_data as md

FLAGS = None

def main(_):
  # Import data
  # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=g.y_, logits=g.y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(g.y_conv, 1), tf.argmax(g.y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  acc_summ = tf.summary.scalar('accuracy', accuracy)

  global_step = tf.Variable(-1, name='global_step')

  sess = tf.InteractiveSession()
  # Store all variables' values now. We want each run to be the same, not random.
  tf.global_variables_initializer().run()

  writer = tf.summary.FileWriter('./log')

  batch_xs = [trainset.draw_class1().reshape(784), trainset.draw_class2().reshape(784)]
  batch_xs = np.array(batch_xs)
  batch_ys = [np.zeros((10)), np.zeros((10))]
  batch_ys[0][7 - 1] = 1
  batch_ys[1][4 - 1] = 1
  batch_ys = np.array(batch_ys)

  # Force conv1 to see features exactly. Diamond, cross, hor, ver.
  value = sess.run(g.W_conv1)
  per_kernel = np.transpose(value, (3, 2, 0, 1))
  per_kernel[0][0] = trainset.draw_cross()
  per_kernel[1][0] = trainset.draw_diamond()
  per_kernel[2][0] = trainset.draw_hor()
  per_kernel[3][0] = trainset.draw_ver()
  per_kernel = tf.transpose(per_kernel, (2, 3, 1, 0))
  sess.run(tf.assign(g.W_conv1, per_kernel))

  # Set conv2 too.
  value = sess.run(g.W_conv2)
  per_kernel = np.transpose(value, (2, 3, 0, 1))
  blank = np.full((4, 4), -1, np.float32)
  # Firstly, from neuron that detects crosses.
  kernel = blank.copy()
  for i in range(4):
    kernel[0, i] = kernel[1, i] = 1
  per_kernel[0][0] = kernel
  per_kernel[0][1] = blank.copy()
  # Secondly, from neuron that detects diamonds.
  per_kernel[1][0] = blank.copy()
  per_kernel[1][1] = blank.copy()
  # Thirdly, from neuron that detects hors.
  per_kernel[2][0] = blank.copy()
  per_kernel[2][1] = blank.copy()
  # Lastly, from neuron that detects vers.
  kernel = blank.copy()
  for i in range(4):
    kernel[2, i] = kernel[3, i] = 1
  per_kernel[3][0] = blank.copy()
  per_kernel[3][1] = kernel
  per_kernel = tf.transpose(per_kernel, (2, 3, 0, 1))
  sess.run(tf.assign(g.W_conv2, per_kernel))

  def train():
    for i in range(10000):
      sess.run(train_step, feed_dict={g.x: batch_xs, g.y_: batch_ys, g.keep_prob:1.0})
      acc_val = sess.run(accuracy,
        feed_dict={g.x: batch_xs, g.y_: batch_ys, g.keep_prob: 1.0})
      print('Train step {}'.format(i, acc_val), end='\r')
      if acc_val == 1:
        print('\n')
        break

    acc_val = sess.run(accuracy,
      feed_dict={g.x: batch_xs, g.y_: batch_ys, g.keep_prob: 1.0})

  def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,784],order='F'),keep_prob:1.0})
    plotNNFilter(units)

  def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
    plt.show()

  def visualize():
    # units = sess.run(g.h_conv1,
    #   feed_dict={g.x: [batch_xs[0]], g.y_: [batch_ys[0]], g.keep_prob: 1.0})
    # plotNNFilter(units)

    print('Input type 1: 7th category')
    print(sess.run(g.y_conv,
      feed_dict={g.x: [batch_xs[0]], g.y_: [batch_ys[0]], g.keep_prob: 1.0}))
    print('Input type 2: 4th category')
    print(sess.run(g.y_conv,
      feed_dict={g.x: [batch_xs[1]], g.y_: [batch_ys[1]], g.keep_prob: 1.0}))

    # Draw inputs
    input = tf.constant(np.array([trainset.draw_class1()]))
    input = tf.expand_dims(input, 3)

    input = tf.summary.image('inputs/class1', input)
    writer.add_summary(sess.run(input))

    input = tf.constant(np.array([trainset.draw_class2()]))
    input = tf.expand_dims(input, 3)

    input = tf.summary.image('inputs/class2', input)
    writer.add_summary(sess.run(input))

    x = utils.kernel_on_grid(g.W_conv1, 5)
    w = tf.transpose(g.W_conv2, (2, 0, 1, 3))

    # Visualize conv1 kernels
    filter_summ = tf.summary.image('conv1/kernels', x, max_outputs=1)
    writer.add_summary(sess.run(filter_summ))
    for i in range(g.num_conv1):
      y = utils.kernel_on_grid(tf.expand_dims(w[i], 2), 5)
      filter_summ = tf.summary.image('conv2/kernels' + str(i), y, max_outputs=1)
      writer.add_summary(sess.run(filter_summ))
    writer.flush()

  train()
  visualize()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='../MNIST-data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  FLAGS.data_aug = True
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
