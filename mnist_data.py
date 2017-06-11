from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

import numpy as np
import elastic_deform as ed

def load_mnist_data(train_dir, one_hot=True):
  """Returns all 'train' data --- images and labels."""
  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  SOURCE_URL = mnist.SOURCE_URL

  local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                   SOURCE_URL + TRAIN_IMAGES)
  with open(local_file, 'rb') as f:
    train_images = mnist.extract_images(f)

  local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                   SOURCE_URL + TRAIN_LABELS)
  with open(local_file, 'rb') as f:
    train_labels = mnist.extract_labels(f, one_hot=one_hot)

  return train_images, train_labels

def select_data(n=10, expand_with_deform=0, train_dir='MNIST-data'):
  """Extracts a subset of mnist train data.
  If doublt_with_deform is True, dataset size is doubled adding a deformed duplicate.
  n is number of examples for each digit/class.

  return normal_dataset, expanded_dataset
  """
  # The 2 datasets to be constructed.
  normal = None
  expanded = None

  train_images, train_labels = load_mnist_data(train_dir)

  numbers = [[] for i in range(10)] # 10 classes, 10 arrays.

  # Take n datapoints for each number.
  for i in range(len(train_labels)):
    if sum(len(x) for x in numbers) == 10 * n:
      break

    number = np.where(train_labels[i] == 1)[0][0]
    if len(numbers[number]) < n:
      numbers[number].append(i)

  # import vis_mnist as vm
  # for i in range(10):
  #   vm.show_image(numbers[i])

  # Scramble subset. 'numbers' contain indices into train_labels.
  numbers = np.asarray(numbers)
  numbers = numbers.reshape(10 * n)
  np.random.shuffle(numbers)

  # Actually retrieve the subset.
  subset_images = []
  subset_labels = []
  for i in numbers:
    subset_images.append(train_images[i])
    subset_labels.append(train_labels[i])

  options = dict(dtype=dtypes.float32, reshape=True, seed=None)
  # Construct normal dataset
  normal = mnist.DataSet(np.asarray(subset_images), np.asarray(subset_labels), **options)

  for j in range(expand_with_deform):
    print("Deforming all 'train' images..")
    count = 0
    for i in numbers:
      shape = train_images[i].shape
      image = train_images[i].reshape((28, 28))
      new_image = ed.rotate(image)
      new_image = ed.translate(new_image)
      new_image = ed.deform(new_image)
      subset_images.append(new_image.reshape(shape))
      subset_labels.append(train_labels[i])
      count += 1
      print('Processed image {}'.format(count), end='\r')
    print("\nDeformation done.")

  subset_images = np.asarray(subset_images)
  subset_labels = np.asarray(subset_labels)

  # Shuffle expanded set.
  perm = np.arange(len(subset_images))
  np.random.shuffle(perm)
  subset_images = subset_images[perm]
  subset_labels = subset_labels[perm]

  expanded = mnist.DataSet(subset_images, subset_labels, **options)
  return normal, expanded

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
