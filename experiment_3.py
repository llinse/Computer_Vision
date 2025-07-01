import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

def rec(src):
    w,h = np.shape(src)
    flt = np.ones(np.shape(src))
    flt1 = np.zeros(np.shape(src))
    rx1 = w / 2
    ry1 = h / 2
    for i in range(1,w):
        for j in range(1,h):
            if abs(j-ry1) <= 5 and abs(i-rx1)>=20:
                flt1[i,j] = 1
    flt=flt-flt1
    return flt,flt1

def dftfilt(f, H):
    F = np.fft.fft2(f)
    g = np.real(np.fft.ifft2(np.multiply(H, F)))
    g = g[g.shape[0]+1:0:-1, g.shape[1]+1:0:-1]
    return g

def dft(img):
    dft = np.fft.fft2(img)
    dftshift = np.fft.fftshift(dft)
    result = np.log(np.abs(dftshift)+1)
    return result

def lvbo(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    dftshift = np.fft.fftshift(np.fft.fft2(img))
    result = dft(img)
    H,H2 = rec(dftshift)
    result2 = np.fft.fftshift(H)
    result3 = dftfilt(img, result2)
    result2_2 = np.fft.fftshift(H2)
    result3_2 = dftfilt(img,result2_2)
    plt.subplot(321), plt.imshow(result, cmap='gray')
    plt.subplot(322), plt.imshow(img, cmap='gray')
    plt.subplot(323), plt.imshow(H, cmap='gray')
    plt.subplot(324), plt.imshow(result3, cmap='gray')
    plt.subplot(325), plt.imshow(H2, cmap='gray')
    plt.subplot(326), plt.imshow(result3_2, cmap='gray')
    plt.show()

lvbo('exam3.tif')
img = cv2.imread('b1.jpg')
cv2.imshow('b1', img)
cv2.waitKey(0)