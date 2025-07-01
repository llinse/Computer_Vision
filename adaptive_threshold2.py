import cv2
import numpy as np
from scipy import signal


img = cv2.imread('bz.jpg', 0)
img = cv2.resize(img, (700, 600))


def adaptive_thres(img, win=3, beta=0.9):
    if win % 2 == 0: win = win - 1
    # 边界的均值有点麻烦
    # 这里分别计算和和邻居数再相除
    kern = np.ones([win, win])
    sums = signal.correlate2d(img, kern, 'same')
    cnts = signal.correlate2d(np.ones_like(img), kern, 'same')
    means = sums // cnts
    # 如果直接采用均值作为阈值，背景会变花
    # 但是相邻背景颜色相差不大
    # 所以乘个系数把它们过滤掉
    img = np.where(img < means * beta, 0, 255)
    return img


img = np.array(img)
print(img.shape)
img = adaptive_thres(img)
img = np.array(img, dtype=np.uint8)
cv2.imshow('deal_image', img)
cv2.waitKey(0)

