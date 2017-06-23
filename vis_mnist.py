import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

import matplotlib.pyplot as plt

def pad(image, padding=1, color=0.5):
  # Border of pixels.
  v_border = np.ndarray((28, padding))
  h_border = np.ndarray((padding, 28 + (padding * 2)))
  v_border.fill(color)
  h_border.fill(color)

  new_img = image
  new_img = np.concatenate((v_border, new_img), axis=1)
  new_img = np.concatenate((new_img, v_border), axis=1)
  new_img = np.concatenate((h_border, new_img))
  new_img = np.concatenate((new_img, h_border))
  return new_img

def show_images(indices):

  input_data_dir= 'MNIST-data'

  data_sets = input_data.read_data_sets(input_data_dir, fake_data=False)

  images = data_sets.train.images

  col = None
  im = None
  for i in range(len(indices)):
    next_image = images[indices[i],:].reshape((28, 28))
    next_image = pad(next_image)
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
    blank = pad(np.zeros((28, 28)))
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
  show_images([0, 1, 2, 3, 4, 5, 6, 7])
