from math import *
from scipy import misc, ndimage
import skimage.transform as st
import cv2
import numpy as np
from scipy import signal

def image_input(img):
    ## 图片二值化、调整大小、高斯滤波
    height, width = img.shape[:2]
    size = (450, 600)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

def image_gau(img):
    paper = cv2.GaussianBlur(img, (3,3), 0)
    # cv2.imshow("paper", paper)
    # cv2.waitKey()
    return paper

def image_ero(paper):
    ## 膨胀
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(paper, kernel)
    # cv2.imshow("erosion", erosion)
    # cv2.waitKey()
    return erosion

def image_canny(erosion):
    ## 边缘检测
    canny = cv2.Canny(erosion, 50,170)
    # cv2.imshow("canny", canny)
    # cv2.waitKey()
    return canny

def image_contour(canny):
    ## 通过求最大面积轮廓，确定纸张轮廓
    contours= cv2.findContours(canny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    max_area = 0
    index = 0
    for i in range(len(contours[1])):
        tmparea = cv2.contourArea(contours[1][i])
        if tmparea > max_area:
            index = i
            max_area = tmparea
    # print(contours[1][index])
    maxcon = []
    conx = []
    cony = []
    gpaper = np.zeros(canny.shape)
    for i in range(len(contours[1][index])):
        maxcon.append(list(contours[1][index][i][0]))
        conx.append(contours[1][index][i][0][1])
        cony.append(contours[1][index][i][0][0])
        gpaper[contours[1][index][i][0][1],contours[1][index][i][0][0]]=1
    # cv2.imshow("can", gpaper)
    # cv2.waitKey()
    return gpaper

def image_rot1(gpaper,paper):
    ## 确定四个端点+透视变换
    lines = st.probabilistic_hough_line(gpaper, threshold=10, line_length=200,line_gap=30)
    gline = gpaper * 0
    line0=[]
    line1=[]
    leftup=0
    leftdown=0
    rightup=0
    rightdown=0
    for line in lines:
        p0, p1 = line
        line0.append(p0)
        line1.append(p1)
        gline[p0[0], p1[0]]=1
        gline[p0[1], p1[1]]=1
        if p0[0]<= 200 and p0[1]<=200:
            leftup = np.array(p0)
        if p1[0]<= 200 and p1[1]<=200:
            leftup = np.array(p1)
        if p0[0]<= 200 and p0[1]>=300:
            leftdown = np.array(p0)
        if p1[0]<= 200 and p1[1]>=300:
            leftdown = np.array(p1)
        if p0[0] >= 250 and p0[1]<=200:
            rightup = np.array(p0)
        if p1[0] >= 250 and p1[1]<=200:
            rightup = np.array(p1)
        if p0[0] >= 250 and p0[1]>=300:
            rightdown = np.array(p0)
        if p1[0] >= 250 and p1[1]>=300:
            rightdown = np.array(p1)
    # print(leftup,leftdown,rightup,rightdown)
    leftup =leftup -30
    leftdown[0]=leftdown[0] -30
    leftdown[1]=leftdown[1] + 90
    rightup[0]=rightup[0] -30
    rightup[1]=rightup[1]
    rightdown[0] =rightdown[0] - 50
    rightdown[1] =rightdown[1]+ 90

    h,w = gpaper.shape
    pts = np.float32([leftup,rightup,leftdown,rightdown])
    pts1 = np.float32([[0,0], [w-1,0], [0,h-1], [w-1, h-1]])
    M = cv2.getPerspectiveTransform(pts, pts1)
    dst = cv2.warpPerspective(paper, M, gpaper.shape)
    # cv2.imshow("M", dst)
    # cv2.waitKey()
    return dst

def image_rot2(canny,paper):
    ## 实现方法二
    lines = cv2.HoughLines(canny,1,np.pi/180,0)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    t = float(y2-y1)/(x2-x1)
    rotate_angle = degrees(atan(t))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
    rotate_img = ndimage.rotate(paper, rotate_angle)
    return rotate_img

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

## 图形矫正
img = cv2.imread("bz.jpg", 0)
img1 = image_input(img)
paper = image_gau(img1)
erosion = image_ero(paper)
canny = image_canny(erosion)
# gpaper = image_contour(canny)
# rot1 = image_rot1(gpaper,img1)
rot2 = image_rot2(canny,img1)

## 图形处理
img2 = np.array(rot2)
print(img2.shape)
img2 = adaptive_thres(img2)
img2 = np.array(img2, dtype=np.uint8)

cv2.imshow('deal_image', img2)
cv2.imshow('paper' , paper)
# cv2.imshow('rot1', rot1)
cv2.imshow('rot2', rot2)
cv2.waitKey(0)



