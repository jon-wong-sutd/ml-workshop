import matplotlib.pyplot as plt
import numpy as np

def draw_seven():
  img = np.ndarray((28, 28))
  img.fill(0)

  img[3, 3] = 0.25
  img[4, 3] = 0.50
  img[5, 3] = 0.25
  for j in range(4, 25):
    img[3, j] = 0.5
    img[4, j] = 1.0
    img[5, j] = 0.5
  img[3, 25] = 0.25
  img[4, 25] = 0.50
  img[5, 25] = 0.25

  for i in range(6, 25):
    startX = 21 - i + 6
    img[i, startX] = 0.25
    img[i, startX + 1] = 0.50
    img[i, startX + 2] = 1
    img[i, startX + 3] = 0.50
    img[i, startX + 4] = 0.25

  return img

def draw_four():
  img = np.ndarray((28, 28))
  img.fill(0)

  for i in range(3, 15):
    startX = 17 - i + 1
    img[i, startX] += 0.25
    img[i, startX + 1] += 0.50
    img[i, startX + 2] += 1
    img[i, startX + 3] += 0.50
    img[i, startX + 4] += 0.25

  startX = 15
  for i in range(3, 25):
    img[i, startX] = 0.25
    img[i, startX + 1] = 0.50
    img[i, startX + 2] = 1.00
    img[i, startX + 3] = 0.50
    img[i, startX + 4] = 0.25
  img[3, startX] = 0
  img[3, startX + 1] = 0.25
  img[3, startX + 2] = 0.50
  img[3, startX + 3] = 0.25
  img[3, startX + 4] = 0
  img[24, startX] = 0
  img[24, startX + 1] = 0.25
  img[24, startX + 2] = 0.50
  img[24, startX + 3] = 0.25
  img[24, startX + 4] = 0

  startY = 15
  startX = 5
  for j in range(startX, startX + 18):
    img[startY, j] = 0.5
    img[startY + 1, j] = 1
    img[startY + 2, j] = 0.5
  img[startY, startX - 1] = 0.25
  img[startY + 1, startX - 1] = 0.5
  img[startY + 2, startX - 1] = 0.25
  img[startY, startX + 18] = 0.25
  img[startY + 1, startX + 18] = 0.5
  img[startY + 2, startX + 18] = 0.25

  return img

def draw_cross(for_neuron=False):
  img = np.ndarray((7, 7), dtype=np.float32)
  img.fill(0)
  if for_neuron is True:
    img.fill(-1)

  for i in range(5):
    img[i + 1, i + 1] = 1
    img[5 - i, i + 1] = 1

  return img

def draw_hor(for_neuron=False):
  img = np.ndarray((7, 7), dtype=np.float32)
  img.fill(0)
  if for_neuron is True:
    img.fill(-1)

  for i in range(2, 5):
    img[1, i] = 1
    img[5, i] = 1

  return img

def draw_ver(for_neuron=False):
  img = np.ndarray((7, 7), dtype=np.float32)
  img.fill(0)
  if for_neuron is True:
    img.fill(-1)

  for i in range(2, 5):
    img[i, 1] = 1
    img[i, 5] = 1

  return img

def draw_diamond(for_neuron=False):
  img = np.ndarray((7, 7), dtype=np.float32)
  img.fill(0)
  if for_neuron is True:
    img.fill(-1)

  img[2, 3] = img[3, 2] = img[3, 4] = img[4, 3] = 1
  return img

def draw_class1():
  row1 = np.concatenate((draw_cross(), draw_cross()), axis=1)
  row1 = np.concatenate((row1, row1), axis=1)
  row1 = np.concatenate((row1, row1))
  # row2 = np.concatenate((draw_diamond(), draw_diamond()), axis=1)
  # row2 = np.concatenate((row2, row2), axis=1)
  # row2 = np.concatenate((row2, row2))
  row2 = np.ndarray((14, 28), dtype=np.float32)
  row2.fill(0)
  img = np.concatenate((row1, row2))
  # img = np.concatenate((row1, row1))
  return img

def draw_class2():
  row1 = np.ndarray((14, 28), dtype=np.float32)
  row1.fill(0)
  # row1 = np.concatenate((draw_hor(), draw_hor()), axis=1)
  # row1 = np.concatenate((row1, row1), axis=1)
  # row1 = np.concatenate((row1, row1))
  row2 = np.concatenate((draw_ver(), draw_ver()), axis=1)
  row2 = np.concatenate((row2, row2), axis=1)
  row2 = np.concatenate((row2, row2))
  img = np.concatenate((row1, row2))
  # img = np.concatenate((row2, row2))
  return img

def test_numbers():
  import sys
  sys.path.insert(0, '../')
  import vis_mnist as vm
  seven = draw_seven();
  seven = vm.pad(seven)
  four = draw_four()
  four = vm.pad(four)

  plt.imshow(seven, cmap='gray')
  plt.show()
  plt.imshow(four, cmap='gray')
  plt.show()

if __name__ == '__main__':
  img = draw_class1()
  plt.imshow(img, cmap='gray')
  plt.show()

  img = draw_class2()
  plt.imshow(img, cmap='gray')
  plt.show()

  # img = draw_diamond()
  # plt.imshow(img, cmap='gray')
  # plt.show()

  # img = draw_hor()
  # plt.imshow(img, cmap='gray')
  # plt.show()

  # img = draw_ver()
  # plt.imshow(img, cmap='gray')
  # plt.show()

  # img = draw_cross()
  # plt.imshow(img, cmap='gray')
  # plt.show()
