import numpy as np
import cv2

# 图片的读取和保存
imgpath = input('请输入图片路径:')
imgpath = str(imgpath)
img = cv2.imread(imgpath,1)
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27:          # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'):  # wait for 's' key to save and exit
    cv2.imwrite('first.png',img)
    cv2.destroyAllWindows()

# 图片平移
rows, cols = img.shape[:2]
x = input('请输入x轴方向平移数值：')
y = input('请输入y轴方向平移数值：')
M = np.float32([[1, 0, x], [0, 1, y]])
imgt = cv2.warpAffine(img, M, (cols, rows)) # 用仿射变换实现平移，图像大小不变

# 图片缩放
xm = input('请输入x轴方向缩放比例：')
ym = input('请输入y轴方向缩放比例：')
xm = float(xm)
ym = float(ym)
imgz = cv2.resize(img,None,fx=xm,fy=ym,interpolation=cv2.INTER_CUBIC)
cv2.imshow('image',imgz)
k = cv2.waitKey(0)

# 图片旋转
rows,cols = img.shape[:2]
rad = input('请输入图片旋转角度：')
M = cv2.getRotationMatrix2D((cols/2, rows/2),rad,1) # 围绕图片中心点，旋转指定角度
imgr = cv2.warpAffine(img, M, (cols, rows))

# 鼠标响应
def getposBgr(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print("Bgr is", img[y, x])
cv2.imshow('image',img)
cv2.setMouseCallback("image", getposBgr)
cv2.waitKey(0)