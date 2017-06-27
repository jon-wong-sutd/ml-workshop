from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

import tensorflow as tf

sys.path.insert(0, '../')
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

  max_acc_summ = tf.summary.scalar('max_accuracy', accuracy)

  sess = tf.InteractiveSession()
  # Store all variables' values now. We want each run to be the same, not random.
  tf.global_variables_initializer().run()
  saver = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2])
  init_save = 'saves/init_values'
  saver.save(sess, init_save)

  normal_writer = tf.summary.FileWriter('./log/normal')
  expanded_writer = tf.summary.FileWriter('./log/expanded')

  def train(train_steps, batch_size, normal_dataset, expanded_dataset):
    # Initialize all variables first.
    saver.restore(sess, init_save)

    checkpoint_step = 100

    def steps(dataset, label, writer):
      highest_acc = 0
      highest_acc_i = 0
      for i in range(train_steps):
        batch_xs, batch_ys = dataset.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob:0.5})
        if i > 0 and i % checkpoint_step == 0:
          # Test trained model
          summ, max_summ, acc_val = sess.run([acc_summ, max_acc_summ, accuracy],
                     feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
          print('Accuracy {} ({}): {}'.format(i, label, acc_val))
          writer.add_summary(summ, i)
          if acc_val > highest_acc:
            writer.add_summary(max_summ, i)
            highest_acc = acc_val
            highest_acc_i = i
          # If more than 5 checkpoints without improvement, quit.
          if i - highest_acc_i >= 5 * checkpoint_step:
            break

    steps(expanded_dataset, 'expanded', expanded_writer)
    saver.restore(sess, init_save)
    steps(normal_dataset, 'normal', normal_writer)

  n = 10
  # Expand normal set by duplicating 5 times. 6 times of normal set size in total.
  normal_dataset, expanded_dataset = md.select_data(n, 5, FLAGS.data_dir)
  # Normal: Highest 0.7601 at step 990. Seems to peak at steps 600-700.
  # Expanded: No peak seen even at step 1000.
  train_steps = 5000
  print('train_steps: {}'.format(train_steps))
  batch_size = 50
  train(train_steps, batch_size, normal_dataset, expanded_dataset)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='../MNIST-data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  FLAGS.data_aug = True
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
