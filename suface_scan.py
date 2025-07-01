import numpy as np
from scipy import signal
import cv2


def order_points(pts):
    # 参考ref[1]
    # 初始化坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 获取左上角和右下角坐标点
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 分别计算左上角和右下角的离散差值
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    # 参考ref[1]
    # 获取坐标点，并将它们分离开来
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算新图片的宽度值，选取水平差值的最大值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 计算新图片的高度值，选取垂直差值的最大值
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 构建新图片的4个坐标点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 获取仿射变换矩阵并应用它
    M = cv2.getPerspectiveTransform(rect, dst)
    # 进行仿射变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后的结果
    return warped


def on_mouse(event, x, y, flags, param):
    global timg, points
    img2 = timg.copy()
    point0 = (0, 0)
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (x, y)
        points.append([x, y])
        print(x, y)
        cv2.circle(img2, point1, 4, (0, 255, 0), 4)
        cv2.imshow('origin', img2)
    return point0


if __name__ == '__main__':
    global points, timg
    xscale, yscale = 0.5, 0.5  # 通过放大图像来使得点击位置更加精确
    points = []
    oimg = cv2.imread('b1.jpg')
    oshape = oimg.shape
    timg = cv2.resize(oimg, (int(oshape[1] / xscale), int(oshape[0] / yscale)))  # 放大图像
    print(timg.shape)
    cv2.imshow('origin', timg)
    cv2.setMouseCallback('origin', on_mouse)
    cv2.waitKey(0)  # 点完4个角点之后随便按一个键盘按键结束操作
    cv2.destroyAllWindows()
    points = np.array(points, dtype=np.float32)
    points[:, 0] *= oshape[1] / int(oshape[1] / xscale)  # 还原像素位置的大小
    points[:, 1] *= oshape[0] / int(oshape[0] / yscale)
    warped = four_point_transform(oimg, points)
    # cv2.imwrite('warped.png', warped ) # 保存图像
    cv2.imshow('origin', warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# img = cv2.imread('b1.jpg',0)
# # img = resize(img)
# # cv2.imshow('b1', img)
# # cv2.waitKey(0)
# img = avr(img)
# cv2.imshow('b', img)
# cv2.waitKey(0)
