import cv2
import sys
import os
import PySide2
import skimage.transform as st
import numpy as np
from scipy import misc, ndimage
from math import *
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import QWidget, QLabel, QPushButton, QApplication,QSpinBox,QGridLayout,QFileDialog

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


class MainApp(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.setup_ui()

    def setup_ui(self):
        self.image_label = QLabel()
        self.image_label1 = QLabel()
        self.image_label2 = QLabel()
        self.image_label3 = QLabel()

        self.rbutton = QPushButton("读取")
        self.rbutton.clicked.connect(self.display)
        self.tbutton = QPushButton("图像处理一")
        self.tbutton.clicked.connect(self.image_process)
        self.pbutton = QPushButton("旋转")
        self.pbutton.clicked.connect(self.romate1)
        self.obutton = QPushButton("图像处理二")
        self.obutton.clicked.connect(self.image_process1)
        self.lbutton = QPushButton("旋转")
        self.lbutton.clicked.connect(self.romate2)
        self.ebutton = QPushButton("图像处理三")
        self.ebutton.clicked.connect(self.image_process2)

        self.main_layout = QGridLayout()
        self.image_layout = QGridLayout()
        self.main_layout.addWidget(self.rbutton,1,1)
        self.main_layout.addWidget(self.tbutton,1,2)
        self.main_layout.addWidget(self.pbutton,2,2)
        self.main_layout.addWidget(self.obutton,1,3)
        self.main_layout.addWidget(self.lbutton,2,3)
        self.main_layout.addWidget(self.ebutton,1,4)
        # self.main_layout.addWidget(self.image_label,2,1)
        # self.main_layout.addWidget(self.image_label1,2,2)
        # self.main_layout.addWidget(self.image_label2,2,3)
        self.image_layout.addWidget(self.image_label,1,1)
        self.image_layout.addWidget(self.image_label1,1,2)
        self.image_layout.addWidget(self.image_label2,1,3)
        self.image_layout.addWidget(self.image_label3,1,4)
        self.image_layout.addLayout(self.main_layout,2,1,1,4)
        self.setLayout(self.image_layout)

    def display(self):
        filePath, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "选择你要上传的图片",  # 标题
            r"C:/Users/xuxh/Desktop/img",  # 起始目录
            "图片类型 (*.png *.jpg *.bmp)"  # 选择类型过滤项，过滤内容在括号中
        )
        self.img1 = cv2.imread(filePath, 0)
        self.img = cv2.imread(filePath, 0)
        # self.img = cv2.imread('b1.jpg', 0)
        size = (450, 600)
        self.img = cv2.resize(self.img, size, interpolation=cv2.INTER_AREA)
        frame = self.img
        image = QImage(frame.data,frame.shape[1], frame.shape[0],
                         frame.shape[1], QImage.Format_Grayscale8)
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def image_process(self):
        paper = cv2.GaussianBlur(self.img, (3,3), 0)
        ## 膨胀
        kernel = np.ones((5,5), np.uint8)
        erosion = cv2.erode(paper, kernel)

        ## 边缘检测
        canny = cv2.Canny(erosion, 50, 170)
        # cv2.imshow('can',canny)

        image, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]  # 计算最大轮廓的旋转包围盒
        rect = cv2.minAreaRect(c)  # 获取包围盒（中心点，宽高，旋转角度）
        box = np.int0(cv2.boxPoints(rect))  # box
        draw_img = cv2.drawContours(self.img.copy(), [box], -1, (0, 0, 255), 3)
        # cv2.imshow('draw',draw_img)
        # cv2.waitKey(0)

        leftup = box[2]
        leftdown = box[1]
        rightup = box[3]
        rightdown = box[0]
        h, w = canny.shape
        pts = np.float32([leftup, rightup, leftdown, rightdown])
        pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        M = cv2.getPerspectiveTransform(pts, pts1)

        dst = cv2.warpPerspective(self.img, M, (canny.shape[1],canny.shape[0]))

        I=dst
        winSize = (5,5)
        ratio = 0.06
        # 第一步:对图像矩阵进行均值平滑
        I_mean = cv2.boxFilter(I, cv2.CV_32FC1, winSize)
        # 第二步:原图像矩阵与平滑结果做差
        out = I - (1.0 - ratio) * I_mean
        # 第三步:当差值大于或等于0时，输出值为255；反之，输出值为0
        out[out >= 0] = 255
        out[out < 0] = 0
        out = out.astype(np.uint8)
        self.out1 = out



        frame = out
        image = QImage(frame.data,frame.shape[1], frame.shape[0],
                         frame.shape[1], QImage.Format_Grayscale8)
        self.image_label1.setPixmap(QPixmap.fromImage(image))


    def romate1(self):
        rows, cols = self.img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)  # 围绕图片中心点，旋转指定角度
        self.out1 = cv2.warpAffine(self.out1, M, (cols, rows))
        frame = self.out1
        image = QImage(frame.data,frame.shape[1], frame.shape[0],
                         frame.shape[1], QImage.Format_Grayscale8)
        self.image_label1.setPixmap(QPixmap.fromImage(image))


    def image_process1(self):
        original_img = self.img
        blurred = cv2.GaussianBlur(original_img, (3, 3), 0)  # 高斯模糊去噪（设定卷积核大小影响效果）
        ret, th2 = cv2.threshold(blurred, 0, 256, cv2.THRESH_OTSU)
        print(ret)
        _, RedThresh = cv2.threshold(blurred, ret - 5, 255, cv2.THRESH_BINARY)  # 设定阈值165（阈值影响开闭运算效果）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义矩形结构元素
        closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)  # 闭运算（链接块）
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)  # 开运算（去噪点）

        image,contours,hierarchy = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]  # 计算最大轮廓的旋转包围盒
        rect = cv2.minAreaRect(c)  # 获取包围盒（中心点，宽高，旋转角度）
        box = np.int0(cv2.boxPoints(rect))  # box
        draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)

        orignal_W = ceil(np.sqrt((box[3][1] - box[2][1]) ** 2 + (box[3][0] - box[2][0]) ** 2))
        orignal_H = ceil(np.sqrt((box[3][1] - box[0][1]) ** 2 + (box[3][0] - box[0][0]) ** 2))

        # 原图中的四个顶点,与变换矩阵
        pts1 = np.float32([box[0], box[1], box[2], box[3]])
        pts2 = np.float32(
            [[0, 0], [int(orignal_W + 1), 0], [int(orignal_W + 1), int(orignal_H + 1)], [0, int(orignal_H + 1)]])

        # 生成透视变换矩阵；进行透视变换
        M = cv2.getPerspectiveTransform(pts1, pts2)
        result_img = cv2.warpPerspective(original_img, M, (int(orignal_W + 3), int(orignal_H + 1)))
        dst1 = cv2.warpPerspective(self.img1, M, (opened.shape[1], opened.shape[0]))


        I=result_img
        winSize = (5,5)
        ratio = 0.06
        # 第一步:对图像矩阵进行均值平滑
        I_mean = cv2.boxFilter(I, cv2.CV_32FC1, winSize)
        # 第二步:原图像矩阵与平滑结果做差
        out = I - (1.0 - ratio) * I_mean
        # 第三步:当差值大于或等于0时，输出值为255；反之，输出值为0
        out[out >= 0] = 255
        out[out < 0] = 0
        out = out.astype(np.uint8)

        I1=dst1
        I_mean1 = cv2.boxFilter(I1, cv2.CV_32FC1, winSize)
        out1 = I1 - (1.0 - ratio) * I_mean1
        out1[out1 >= 0] = 255
        out1[out1 < 0] = 0
        out1 = out1.astype(np.uint8)

        self.out2=out
        frame = out
        image = QImage(frame.data,frame.shape[1], frame.shape[0],
                         frame.shape[1], QImage.Format_Grayscale8)
        self.image_label1.setPixmap(QPixmap.fromImage(image))

        cv2.imwrite('first.png', out1)

    def romate2(self):
        rows, cols = self.img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)  # 围绕图片中心点，旋转指定角度
        self.out2 = cv2.warpAffine(self.out2, M, (cols, rows))
        frame = self.out2
        image = QImage(frame.data,frame.shape[1], frame.shape[0],
                         frame.shape[1], QImage.Format_Grayscale8)
        self.image_label1.setPixmap(QPixmap.fromImage(image))

    def image_process2(self):
        I = self.img
        winSize = (5, 5)
        ratio = 0.06
        # 第一步:对图像矩阵进行均值平滑
        I_mean = cv2.boxFilter(I, cv2.CV_32FC1, winSize)
        # 第二步:原图像矩阵与平滑结果做差
        out = I - (1.0 - ratio) * I_mean
        # 第三步:当差值大于或等于0时，输出值为255；反之，输出值为0
        out[out >= 0] = 255
        out[out < 0] = 0
        out = out.astype(np.uint8)

        self.out2 = out
        frame = out
        image = QImage(frame.data, frame.shape[1], frame.shape[0],
                       frame.shape[1], QImage.Format_Grayscale8)
        self.image_label1.setPixmap(QPixmap.fromImage(image))




if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())
