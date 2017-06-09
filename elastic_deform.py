import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

import matplotlib.pyplot as plt

def white_border(image):
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

def test():
  """Deform the first element of 'train' data set."""
  input_data_dir= '/tmp/tensorflow/mnist/input_data'

  data_sets = input_data.read_data_sets(input_data_dir, fake_data=False)

  image = data_sets.train.images[0,:].reshape((28, 28))

  im = image

  # sigma = 2
  # alphas = np.arange(0, 10, 10 / 4)
  # for i in np.nditer(alphas):
  #   alpha = np.asscalar(i)
  for i in range(4):
    # deformed = deform(image, alpha, sigma)
    deformed = deform(image)
    im = np.concatenate((im, deformed))

  plt.imshow(im, cmap='gray')

  plt.show()

if __name__=='__main__':
  test()
