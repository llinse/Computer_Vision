import cv2
import numpy as np
from scipy import signal
from skimage import filters
import matplotlib.pyplot as plt

img = cv2.imread('bz.jpg', 0)
img = cv2.resize(img, (700, 600))


def adaptiveThresh(I, winSize, ratio=0.15):
    # 第一步:对图像矩阵进行均值平滑
    I_mean = cv2.boxFilter(I, cv2.CV_32FC1, winSize)
    # 第二步:原图像矩阵与平滑结果做差
    out = I - (1.0 - ratio) * I_mean
    # 第三步:当差值大于或等于0时，输出值为255；反之，输出值为0
    out[out >= 0] = 255
    out[out < 0] = 0
    out = out.astype(np.uint8)
    return out


img = adaptiveThresh(img, (5, 5))

cv2.imshow('deal_image', img)
cv2.waitKey(0)
plt.show()
