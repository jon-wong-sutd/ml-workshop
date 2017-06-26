import graph as g
import trainset_2elems as trainset
import numpy as np

import tensorflow as tf

def set_conv1(sess, deactivate=False):
  value = sess.run(g.W_conv1)
  per_kernel = np.transpose(value, (3, 2, 0, 1))
  if deactivate is False:
    per_kernel[0][0] = trainset.draw_cross()
    per_kernel[1][0] = trainset.draw_diamond()
    per_kernel[2][0] = trainset.draw_hor()
    per_kernel[3][0] = trainset.draw_ver()
  else:
    blank = np.full((7, 7), -1, np.float32)
    per_kernel[0][0] = blank.copy()
    per_kernel[1][0] = blank.copy()
    per_kernel[2][0] = blank.copy()
    per_kernel[3][0] = blank.copy()
  per_kernel = tf.transpose(per_kernel, (2, 3, 1, 0))
  sess.run(tf.assign(g.W_conv1, per_kernel))

def set_conv2(sess, deactivate=False, corrupt=False):
  value = sess.run(g.W_conv2)
  per_kernel = np.transpose(value, (2, 3, 0, 1))
  blank = np.full((4, 4), -1, np.float32)

  # Set all blank first.
  for i in range(4):
    for j in range(2):
      per_kernel[i][j] = blank.copy()

  span = 4
  if corrupt is True:
    span = 2

  if deactivate is False:
    # Firstly, from neuron that detects crosses.
    kernel = blank.copy()
    for i in range(span):
      kernel[0, i] = kernel[1, i] = 1
    per_kernel[0][0] = kernel
    # Secondly, from neuron that detects diamonds.
    # But no data contains diamonds.
    # Thirdly, from neuron that detects hors.
    # But no data contains hors.
    # Lastly, from neuron that detects vers.
    kernel = blank.copy()
    for i in range(span):
      kernel[2, i] = kernel[3, i] = 1
    per_kernel[3][0] = blank.copy()
    per_kernel[3][1] = kernel

  per_kernel = tf.transpose(per_kernel, (2, 3, 0, 1))
  sess.run(tf.assign(g.W_conv2, per_kernel))
