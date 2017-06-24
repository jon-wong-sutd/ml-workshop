import vis_mnist as vm
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

seven = draw_seven();
seven = vm.pad(seven)
four = draw_four()
four = vm.pad(four)

plt.imshow(seven, cmap='gray')
plt.show()
plt.imshow(four, cmap='gray')
plt.show()
