import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import interpolation

import random

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

import matplotlib.pyplot as plt

def white_border(image):
  image = np.copy(image)
  for i in range(1, 27):
    image[i, 0] = image[i, 27] = 1

  image[0, 0:] = image[27, 0:] = 1
  return image

def deform(image, alpha=5, sigma=2, random_state=None):
  """Smaller sigma means more scatter. Larger alpha means more distortion.
  Every call to this produces random result.
  """
  assert len(image.shape)==2

  if random_state is None:
    random_state = np.random.RandomState(None)

  shape = image.shape

  dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
  dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

  x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
  indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

  return map_coordinates(image, indices, order=1).reshape(shape)

def rotate(image, min_angle=10, max_angle=20):
  angle = random.randint(min_angle, max_angle)
  if random.randint(0, 1) == 1:
    angle = -angle
  return interpolation.rotate(image, angle, reshape=False)

def translate(image):
  # Find bounding box.
  rows = np.any(image, axis=1)
  cols = np.any(image, axis=0)
  rmin, rmax = np.where(rows)[0][[0, -1]]
  cmin, cmax = np.where(cols)[0][[0, -1]]

  # Shift between 3-5 pixels in both axes.
  row_shift = random.randint(3, 5)
  if random.randint(0, 1) == 1:
    row_shift = -row_shift
  col_shift = random.randint(3, 5)
  if random.randint(0, 1) == 1:
    col_shift = -col_shift

  # If row_shift is negative, watch that row_shift >= -rmin
  # If row_shift is positive, watch that row_shift <= (27 - rmax)
  # Same for col_shift
  if row_shift < -rmin:
    row_shift = -rmin
  elif row_shift > (27 - rmax):
    row_shift = 27 - rmax
  if col_shift < -cmin:
    col_shift = -cmin
  elif col_shift > (27 - cmax):
    col_shift = 27 - cmax

  return interpolation.shift(image, (row_shift, col_shift))

def rotate_test(image):
  im = white_border(image)

  deformed = white_border(rotate(image))
  im = np.concatenate((im, deformed))

  # for i in range(4):
  #   # deformed = deform(image, alpha, sigma)
  #   # deformed = deform(image)
  #   deformed = rotate(image, (i + 1) * -10)
  #   im = np.concatenate((im, deformed))

  # for i in range(4):
  #   # deformed = deform(image, alpha, sigma)
  #   # deformed = deform(image)
  #   deformed = rotate(image, (i + 1) * 10)
  #   im = np.concatenate((im, deformed))

  return im

def deform_test(image):
  im = white_border(image)

  deformed = white_border(deform(image))
  im = np.concatenate((im, deformed))

  return im

def translate_test(image):
  im = white_border(image)

  deformed = white_border(translate(image))
  im = np.concatenate((im, deformed))

  return im

def test():
  """Deform the first element of 'train' data set."""
  input_data_dir= 'MNIST-data'

  data_sets = input_data.read_data_sets(input_data_dir, fake_data=False)

  image = data_sets.train.images[0,:].reshape((28, 28))

  im = rotate_test(image)
  plt.imshow(im, cmap='gray')
  plt.title('Rotate')
  plt.show()

  im = translate_test(image)
  plt.imshow(im, cmap='gray')
  plt.title('Translate')
  plt.show()

  im = deform_test(image)
  plt.imshow(im, cmap='gray')
  plt.title('Deform')
  plt.show()

if __name__=='__main__':
  test()
