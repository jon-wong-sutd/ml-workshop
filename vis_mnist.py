import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

import matplotlib.pyplot as plt

def white_border(image):
  for i in range(1, 27):
    image[i, 0] = image[i, 27] = 1

  image[0, 0:] = image[27, 0:] = 1
  return image

def show_image(indices):

  input_data_dir= '/tmp/tensorflow/mnist/input_data'

  data_sets = input_data.read_data_sets(input_data_dir, fake_data=False)

  images = data_sets.train.images

  col = None
  im = None
  for i in range(len(indices)):
    next_image = images[indices[i],:].reshape((28, 28))
    next_image = white_border(next_image)
    if col is None:
      col = next_image
    else:
      col = np.concatenate((col, next_image))

    if i > 0 and i % 3 == 2:
      # Column is full. Append to im.
      if im is None:
        im = col
      else:
        im = np.concatenate((im, col), axis=1)
      col = None

  if col is not None:
    blank = np.zeros((28, 28))
    while i % 3 != 2:
      col = np.concatenate((col, blank))
      i += 1

    if im is None:
      im = col
    else:
      im = np.concatenate((im, col), axis=1)

  plt.imshow(im, cmap='gray')

  plt.show()

if __name__=='__main__':
  some_image([0, 1, 2, 3, 4, 5, 6, 7])
  # vissomedigits([0, 1])
