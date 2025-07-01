from math import *
from sympy import *
import cv2
import numpy as np
import copy
import scipy.signal as signal
import pylab as pl

car = cv2.imread("实验4.tif", 0)
car = cv2.GaussianBlur(car, (9,9), 0)
row, column = car.shape
car_f = np.copy(car)/255.0  # 像素值0-1之间

# sobel算子
sobelx = cv2.Sobel(car_f, -1, 1, 0, ksize=3) #只计算x方向
sobely = cv2.Sobel(car_f, -1, 0, 1, ksize=3) #只计算y方向
mag, ang = cv2.cartToPolar(sobelx, sobely, angleInDegrees=1)  # 得到梯度幅度和梯度角度阵列
cv2.imshow('M', mag)
cv2.imshow('x and y', np.hstack((sobelx, sobely)))
# cv2.waitKey(0)

# 步骤二
g1 = np.zeros(car_f.shape)  # g与图片大小相同
X1, Y1 = np.where((mag > np.max(mag) * 0.4) & (mag < np.max(mag) * 0.9) &(ang >= 45)&(ang <= 135) )
X5, Y5 = np.where((mag > np.max(mag) * 0.4) & (mag < np.max(mag) * 0.9) &(ang >= 245)&(ang <= 315) )
g1[X1, Y1] = 1
g1[X5, Y5] = 1

g2 = np.zeros(car_f.shape)  # g与图片大小相同
X2, Y2 = np.where((mag > np.max(mag) * 0.4) & (mag < np.max(mag) * 0.7) & ( ang <= 45) )
X3, Y3 = np.where((mag > np.max(mag) * 0.4) & (mag < np.max(mag) * 0.7) & ( ang >= 135) & (ang <=225))
X4, Y4 = np.where((mag > np.max(mag) * 0.4) & (mag < np.max(mag) * 0.7) & ( ang >= 315))
g2[X2, Y2] = 1
g2[X3, Y3] = 1
g2[X4, Y4] = 1

# 行扫描，间隔k时，进行填充，填充值为1
def edge_connection(img, size, k):
    for i in range(size):
        Yi = np.where(img[i,:] > 0)
        if len(Yi[0]) > 11:  # 可调整
            for j in range(0, len(Yi[0]) - 1):
                if Yi[0][j + 1] - Yi[0][j] <= k:
                    img[i, Yi[0][j]:Yi[0][j + 1]] = 1
    return img


gx = edge_connection(g1, car_f.shape[0], k=25)
g2 = cv2.rotate(g2, 0)
gy = edge_connection(g2, car_f.shape[1], k=25)
gy = cv2.rotate(gy, 2)
g = gx + gy
cv2.imshow("gy", gy)
cv2.imshow("gx", gx)
cv2.imshow("g", g)
# cv2.waitKey(0)

# 形态学细化
def refining(f):
    rows, cols = f.shape
    # 细化模板
    B1 = np.array([-1, -1, -1, 0, 1, 0, 1, 1, 1]).reshape(3, 3)
    B2 = np.array([0, -1, -1, 1, 1, -1, 1, 1, 0]).reshape(3, 3)
    B3 = np.array([1, 0, -1, 1, 1, -1, 1, 0, -1]).reshape(3, 3)
    B4 = np.array([1, 1, 0, 1, 1, -1, 0, -1, -1]).reshape(3, 3)
    B5 = np.array([1, 1, 1, 0, 1, 0, -1, -1, -1]).reshape(3, 3)
    B6 = np.array([0, 1, 1, -1, 1, 1, -1, -1, 0]).reshape(3, 3)
    B7 = np.array([-1, 0, 1, -1, 1, 1, -1, 0, 1]).reshape(3, 3)
    B8 = np.array([-1, -1, 0, -1, 1, 1, 0, 1, 1]).reshape(3, 3)
    maskList = [B1, B2, B3, B4, B5, B6, B7, B8]
    count = 0
    # skemask1 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1]).reshape(3, 3)
    while True:
        temp = f.copy
        for m in maskList:
            mas = []
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if f[i, j] == 0:
                        continue
                    elif np.sum(m * f[i - 1:i + 2, j - 1:j + 2]) == 4:
                        # 击中时标记删除点
                        mas.append((i, j))
            for it in mas:
                x, y = it
                f[x, y] = 0
        if (temp == f).all:
            count += 1
        else:
            count = 0
        if count == 8:
            break
    return f

f = refining(g)
cv2.imshow('f',f)
cv2.waitKey(0)