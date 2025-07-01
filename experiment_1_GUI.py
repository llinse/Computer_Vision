import time
import cv2
import sys
import os
import numpy as np
import PySide2
from PySide2 import QtWidgets
from PySide2.QtCore import QTimer, QSize
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,QApplication,QSpinBox,QGridLayout,QMainWindow

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

class MainApp(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.video_size = QSize(500, 500)
        self.setup_ui()

    def setup_ui(self):
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.video_size)

        self.rbutton = QPushButton("读取")
        self.rbutton.clicked.connect(self.display)
        self.tbutton = QPushButton("平移")
        self.tbutton.clicked.connect(self.trans)
        self.pbutton = QPushButton("旋转")
        self.pbutton.clicked.connect(self.romate)
        self.zbutton = QPushButton("缩放")
        self.zbutton.clicked.connect(self.zoom)
        self.sbutton = QPushButton("保存")
        self.sbutton.clicked.connect(self.save)

        self.text_label = QLabel()
        # poi = cv2.setMouseCallback("image", self.getposBgr)
        # self.text_label.setText(poi)

        self.hbutton = QPushButton("获取")
        self.hbutton.clicked.connect(self.get)

        self.main_layout = QGridLayout()
        self.image_layout = QGridLayout()
        self.main_layout.addWidget(self.rbutton,1,1)
        self.main_layout.addWidget(self.tbutton,1,2)
        self.main_layout.addWidget(self.pbutton,1,3)
        self.main_layout.addWidget(self.zbutton,1,4)
        self.main_layout.addWidget(self.sbutton,1,5)
        self.image_layout.addWidget(self.image_label,1,1,2,1)
        self.image_layout.addWidget(self.hbutton, 1, 2, 1, 1)
        self.image_layout.addWidget(self.text_label, 2,2,1,1)
        self.image_layout.addLayout(self.main_layout,3,1,1,2)
        self.setLayout(self.image_layout)

    def display(self):
        self.img = cv2.imread('a.jpg', 1)
        frame = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        image = QImage(frame, frame.shape[1], frame.shape[0],
                       frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def trans(self):
        # label1 = QLabel('请输入x轴方向平移数值：')
        # label2 = QLabel('请输入y轴方向平移数值：')
        # self.main_layout.addWidget(label1,2,1)
        # self.main_layout.addWidget(label2,3,1)
        self.SpinBox1 = QSpinBox()
        self.SpinBox1.resize(50, 20)
        self.SpinBox1.setRange(0, 1000)
        self.main_layout.addWidget(self.SpinBox1,2,2)
        self.xt=self.SpinBox1.value()

        self.SpinBox2 = QSpinBox()
        self.SpinBox2.resize(50, 20)
        self.SpinBox2.setRange(0, 1000)
        self.main_layout.addWidget(self.SpinBox2,3,2)
        self.yt=self.SpinBox2.value()

        self.buttont = QPushButton("确定")
        self.buttont.clicked.connect(self.trans2)
        self.main_layout.addWidget(self.buttont, 4, 2)

    def trans2(self):
        xt = self.SpinBox1.value()
        yt = self.SpinBox2.value()
        rows, cols = self.img.shape[:2]
        M = np.float32([[1, 0,xt], [0, 1, yt]])
        self.img = cv2.warpAffine(self.img, M, (cols, rows))  # 用仿射变换实现平移，图像大小不变
        frame = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        image = QImage(frame, frame.shape[1], frame.shape[0],
                       frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))
        # self.main_layout.removeWidget(self.SpinBox2)

    def romate(self):
        # label1 = QLabel('请输入图片旋转角度：')
        # self.main_layout.addWidget(label1, 2,2)
        self.SpinBox3 = QSpinBox()
        self.SpinBox3.resize(50, 20)
        self.SpinBox3.setRange(0, 360)
        self.main_layout.addWidget(self.SpinBox3,2,3)
        self.buttonr = QPushButton("确定")
        self.buttonr.clicked.connect(self.romate2)
        self.main_layout.addWidget(self.buttonr,3,3)

    def romate2(self):
        rows, cols = self.img.shape[:2]
        rad = self.SpinBox3.value()
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rad, 1)  # 围绕图片中心点，旋转指定角度
        self.img = cv2.warpAffine(self.img, M, (cols, rows))
        frame = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        image = QImage(frame, frame.shape[1], frame.shape[0],
                       frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def zoom(self):
        # label1 = QLabel('请输入x轴方向缩放比例：')
        # label2 = QLabel('请输入y轴方向缩放比例：')
        # self.main_layout.addWidget(label1, 2, 1)
        # self.main_layout.addWidget(label2, 3, 1)
        self.SpinBox4 = QSpinBox()
        self.SpinBox4.resize(50, 20)
        self.SpinBox4.setRange(0, 100)
        self.main_layout.addWidget(self.SpinBox4, 2, 4)
        self.SpinBox5 = QSpinBox()
        self.SpinBox5.resize(50, 20)
        self.SpinBox5.setRange(0, 100)
        self.main_layout.addWidget(self.SpinBox5, 3, 4)
        self.buttonz = QPushButton("确定")
        self.buttonz.clicked.connect(self.zoom2)
        self.main_layout.addWidget(self.buttonz, 4, 4)

    def zoom2(self):
        xm = self.SpinBox4.value()
        ym = self.SpinBox5.value()
        xm = float(xm)
        ym = float(ym)
        self.img = cv2.resize(self.img, None, fx=xm, fy=ym, interpolation=cv2.INTER_CUBIC)
        print(self.img)
        frame = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        image = QImage(frame, frame.shape[1], frame.shape[0],
                       frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def save(self):
        cv2.imwrite('first.png', self.img)

    def get(self):
        a=cv2.imshow('python',self.img)
        cv2.setMouseCallback('python', self.getposBgr)
        cv2.waitKey(0)

    def getposBgr(self,event, x, y, flags, param):
        while event==cv2.EVENT_LBUTTONDOWN:
            self.text_label.setText("Bgr is"+str([y,x]))
            return self.img[y,x]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())

