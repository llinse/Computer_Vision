from math import *
from scipy import misc, ndimage
import skimage.transform as st
import cv2
import numpy as np
# import matplotlib.pyplot as plt


## 图片二值化、调整大小、高斯滤波
img = cv2.imread("1.jpg", 0)
height, width = img.shape[:2]
size = (450, 600)
img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
paper = cv2.GaussianBlur(img, (3,3), 0)
cv2.imshow("paper", paper)
# cv2.waitKey()

## 膨胀
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(paper, kernel)
cv2.imshow("erosion", erosion)
cv2.waitKey()

## 边缘检测
canny = cv2.Canny(erosion, 50,170)
cv2.imshow("canny", canny/255)
# cv2.waitKey()

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
gpaper = np.zeros(canny.shape)
for i in range(len(contours[1][index])):
    maxcon.append(list(contours[1][index][i][0]))
    gpaper[contours[1][index][i][0][1],contours[1][index][i][0][0]]=1
cv2.imshow("can", gpaper)
cv2.waitKey()

image, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
c = sorted(contours, key=cv2.contourArea, reverse=True)[0]  # 计算最大轮廓的旋转包围盒
rect = cv2.minAreaRect(c)  # 获取包围盒（中心点，宽高，旋转角度）
box = np.int0(cv2.boxPoints(rect))  # box
draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), 3)
cv2.imshow('draw',draw_img)
cv2.waitKey(0)
print(box)
leftup=box[2]
leftdown=box[1]
rightup=box[3]
rightdown=box[0]

# result1 = img.copy()
# lines = cv2.HoughLinesP(canny,1,np.pi/180,50,minLineLength=100,maxLineGap=20)
# print(lines)
# for x1,y1,x2,y2 in lines[5]:
#     cv2.line(result1,(x1,y1),(x2,y2),(0,0,255),1)
# cv2.imshow("result", result1)
# print (x1,y1)
# print (x2,y2)
# cv2.waitKey()

## 透视变换
# h, theta, d = st.hough_line(gpaper)
# h1,theta1,d1 = st.hough_line_peaks(h, theta, d)
# row1, col1 = gpaper.shape
# y0 = (d1 - 0 * np.cos(theta1)) / np.sin(theta1)
# y1 = (d1 - col1 * np.cos(theta1)) / np.sin(theta1)
# plt.plot((0, col1), (y0, y1), '-r')
# plt.axis((0, col1, row1, 0))
# plt.show()
# print(h1)

# result1 = img.copy()
# lines = st.probabilistic_hough_line(gpaper, threshold=10, line_length=200,line_gap=30)
# leftup=0
# leftdown=0
# rightup=0
# rightdown=0
# print(lines)
# for line in lines:
#     p0, p1 = line
#     if p0[0]<= 150 and p0[1]<=150:
#         leftup = np.array(p0)
#     if p1[0]<= 150 and p1[1]<=150:
#         leftup = np.array(p1)
#     if p0[0]<= 200 and p0[1]>=400:
#         leftdown = np.array(p0)
#     if p1[0]<= 200 and p1[1]>=400:
#         leftdown = np.array(p1)
#     if p0[0] >= 350 and p0[1]<=200:
#         rightup = np.array(p0)
#     if p1[0] >= 350 and p1[1]<=200:
#         rightup = np.array(p1)
#     if p0[0] >= 350 and p0[1]>=400:
#         rightdown = np.array(p0)
#     if p1[0] >= 350 and p1[1]>=400:
#         rightdown = np.array(p1)
# print(leftup)
# print(leftdown)
# print(rightup)
# print(rightdown)
# # leftup =leftup -30
# leftdown[0]=leftdown[0] - 30
# leftdown[1]=leftdown[1] + 90
# # rightup[0]=rightup[0] -30
# rightup[1]=rightup[1]
# rightdown[0] =rightdown[0] - 50
# rightdown[1] =rightdown[1]+ 90


h,w = canny.shape
pts = np.float32([leftup,rightup,leftdown,rightdown])
pts1 = np.float32([[0,0], [w-1,0], [0,h-1], [w-1, h-1]])
M = cv2.getPerspectiveTransform(pts, pts1)
dst = cv2.warpPerspective(img, M, canny.shape)
cv2.imshow("M", dst)
cv2.waitKey()


# ## 简便实现
# lines = cv2.HoughLines(canny,1,np.pi/180,0)
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
# t = float(y2-y1)/(x2-x1)
# rotate_angle = degrees(atan(t))
# print(rotate_angle)
# if rotate_angle > 45:
# 	rotate_angle = -90 + rotate_angle
# elif rotate_angle < -45:
# 	rotate_angle = 90 + rotate_angle
# rotate_img = ndimage.rotate(paper, rotate_angle)
# cv2.imshow('rot',rotate_img)
# cv2.waitKey()