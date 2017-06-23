from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

import tensorflow as tf

import graph as g

import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from math import sqrt

def kernel_on_grid(kernel, pad):
  def factorization(n):
      for i in range(int(sqrt(float(n))), 0, -1):
        if n % i == 0:
          if i == 1: print('Who would enter a prime number of filters')
          return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
  print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + 2 * pad
  X = kernel.get_shape()[1] + 2 * pad

  channels = kernel.get_shape()[2]

  # put NumKernels to the 1st dimension
  x = tf.transpose(x, (3, 0, 1, 2))
  # organize grid on Y axis
  x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))
  print(x.shape)

  # switch X and Y axes
  x = tf.transpose(x, (0, 2, 1, 3))
  # organize grid on X axis
  x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x = tf.transpose(x, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x = tf.transpose(x, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return x

def main(_):
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=g.y_, logits=g.y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(g.y_conv, 1), tf.argmax(g.y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  acc_summ = tf.summary.scalar('accuracy', accuracy)

  sess = tf.InteractiveSession()
  saver = tf.train.Saver([g.W_conv1, g.b_conv1, g.W_conv2, g.b_conv2, g.W_fc1, g.b_fc1, g.W_fc2, g.b_fc2])
  save_file = 'mnist_conv/model'

  writer = tf.summary.FileWriter('./log')

  def train():
    # Init highest_acc.
    highest_acc = 0
    file = Path(save_file + '.index')
    if file.is_file():
      saver.restore(sess, save_file)

    x = kernel_on_grid(g.W_conv1, 5)
    w = tf.transpose(g.W_conv2, (2, 0, 1, 3))

    # Visualize conv1 kernels
    filter_summ = tf.summary.image('conv1/kernels', x, max_outputs=1)
    writer.add_summary(sess.run(filter_summ))
    for i in range(32):
      y = kernel_on_grid(tf.expand_dims(w[i], 2), 5)
      filter_summ = tf.summary.image('conv2/kernels' + str(i), y, max_outputs=1)
      writer.add_summary(sess.run(filter_summ))
    writer.flush()

  train()

if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])
