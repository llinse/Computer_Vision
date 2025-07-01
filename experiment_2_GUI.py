import cv2
import sys
import os
import numpy as np
import PySide2
import matplotlib.pyplot as plt
from math import *
from PySide2.QtCore import QTimer, QSize
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,QApplication,QSpinBox,QGridLayout,QMainWindow,QDoubleSpinBox,QGraphicsScene

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

class MainApp(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        self.setup_ui()

    def setup_ui(self):
        self.image_label = QLabel()
        self.image_label2 = QLabel()

        self.rbutton = QPushButton("读取")
        self.rbutton.clicked.connect(self.display)
        self.tbutton = QPushButton("获取全局直方图")
        self.tbutton.clicked.connect(self.image_hist)
        self.pbutton = QPushButton("获取全局均值方差")
        self.pbutton.clicked.connect(self.image_a)
        self.zbutton = QPushButton("获取局部均值方差")
        self.zbutton.clicked.connect(self.image_l)
        self.sbutton = QPushButton("图像增强")
        self.sbutton.clicked.connect(self.stren)

        self.text_label = QLabel()
        self.text2_label = QLabel()

        self.main_layout = QGridLayout()
        self.image_layout = QGridLayout()
        self.main_layout.addWidget(self.rbutton,1,1)
        self.main_layout.addWidget(self.tbutton,1,2)
        self.main_layout.addWidget(self.pbutton,1,3)
        self.main_layout.addWidget(self.zbutton,1,4)
        self.main_layout.addWidget(self.sbutton,1,5,1,2)
        self.image_layout.addWidget(self.image_label,1,1,2,1)
        self.image_layout.addWidget(self.image_label2,1,2,2,1)
        self.image_layout.addWidget(self.text2_label, 2,3, 1, 1)
        self.image_layout.addWidget(self.text_label, 1,3,1,1)
        self.image_layout.addLayout(self.main_layout,3,1,1,3)
        self.setLayout(self.image_layout)

    def display(self):
        self.img = cv2.imread('实验2.tif', 0)
        frame = self.img
        image = QImage(frame.data,frame.shape[1], frame.shape[0],
                         frame.shape[1], QImage.Format_Grayscale8)
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def image_hist(self):
        H = self.img.shape[0]
        W = self.img.shape[1]
        self.hr = np.zeros(256)  # 原始直方图信息
        self.pr = np.zeros(256)  # 原始图片的概率
        for row in range(H):
            for col in range(W):
                self.hr[self.img[row, col]] += 1
        for i in range(256):
            self.pr[i] = self.hr[i] / (H * W)
        plt.plot(self.pr)
        plt.xlim([0, 256])
        plt.show()

    # 全局均值及方差计算
    def image_a(self):
        self.ave = 0
        self.ars = 0
        for i in range(256):
            self.ave += i * self.pr[i]
        for i in range(256):
            self.ars += (i - self.ave) * (i - self.ave) * self.pr[i]
        self.text_label.setText('全局均值为：' + str(self.ave))
        self.text2_label.setText('全局方差为：' + str(self.ars))


    def image_l(self):
        self.SpinBox4 = QSpinBox()
        self.SpinBox4.resize(50, 20)
        self.SpinBox4.setRange(0, 100)
        self.main_layout.addWidget(self.SpinBox4, 2, 4)
        self.buttonz = QPushButton("确定")
        self.buttonz.clicked.connect(self.image_l2)
        self.main_layout.addWidget(self.buttonz, 3, 4)

    def image_l2(self):
        xm = self.SpinBox4.value()
        pad = floor(xm / 2)  # 原图片需要填充的区域
        new_image = np.pad(self.img, ((pad, pad), (pad, pad)), 'constant')  # 填充后的新图片
        self.sigma = np.zeros(self.img.shape)  # 储存局部方差
        self.mean = np.zeros(self.img.shape)  # 储存局部均值
        h = self.img.shape[0]
        w = self.img.shape[1]
        for i in range(abs(h)):
            for j in range(abs(w)):
                sub_domain = new_image[i:i + 2 * pad, j: j + 2 * pad]
                # 局部直方图
                # image_hist(sub_domain)
                element = np.array(sub_domain.flatten())  # 邻域内所有元素
                local_mean = np.mean(element)  # 局部均值
                self.mean[i, j] = local_mean
                # sigma[i, j] = sum((element - local_mean) ** 2) / (size ** 2)   # 局部方差
                self.sigma[i, j] = sqrt(np.var(element))
        mea=np.mean(self.mean)
        sig=np.mean(self.sigma)
        self.text_label.setText('局部均值平均数为：' + str(mea))
        self.text2_label.setText('局部方差平均数为：' + str(sig))
        # return sigma, mean

    def stren(self):
        self.SpinBox0 = QDoubleSpinBox()
        self.SpinBox0.resize(50, 20)
        self.SpinBox0.setRange(0, 100)
        self.main_layout.addWidget(self.SpinBox0, 2, 6)
        label0 = QLabel('k_0值')
        self.main_layout.addWidget(label0, 2, 5)

        self.SpinBox1 = QDoubleSpinBox()
        self.SpinBox1.resize(50, 20)
        self.SpinBox1.setRange(0, 100)
        self.main_layout.addWidget(self.SpinBox1, 3, 6)
        label1 = QLabel('k_1值')
        self.main_layout.addWidget(label1, 3, 5)

        self.SpinBox2 = QDoubleSpinBox()
        self.SpinBox2.resize(50, 20)
        self.SpinBox2.setRange(0, 100)
        self.main_layout.addWidget(self.SpinBox2, 4, 6)
        label2 = QLabel('k_2值')
        self.main_layout.addWidget(label2, 4, 5)

        self.SpinBox3 = QDoubleSpinBox()
        self.SpinBox3.resize(50, 20)
        self.SpinBox3.setRange(0, 100)
        self.main_layout.addWidget(self.SpinBox3, 5, 6)
        label3 = QLabel('E值')
        self.main_layout.addWidget(label3, 5, 5)

        self.buttonse = QPushButton("确定")
        self.buttonse.clicked.connect(self.strengthen)
        self.main_layout.addWidget(self.buttonse, 6, 5,1,2)

    def strengthen(self):
        k_0 = self.SpinBox0.value()
        k_1 = self.SpinBox1.value()
        k_2 = self.SpinBox2.value()
        E = self.SpinBox3.value()
        h = self.img.shape[0]
        w = self.img.shape[1]
        for i in range(abs(h)):
            for j in range(abs(w)):
                if self.mean[i, j] <= k_0 * self.ave:
                    if self.sigma[i, j] <= k_2 * sqrt(self.ars) and self.sigma[i, j] >= k_1 * sqrt(self.ars):
                        self.img[i, j] = E * self.img[i, j]
        frame2 = self.img
        print(frame2)
        image2 = QImage(frame2.data,frame2.shape[1], frame2.shape[0],
                         frame2.shape[1], QImage.Format_Grayscale8)
        self.image_label2.setPixmap(QPixmap.fromImage(image2))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())
