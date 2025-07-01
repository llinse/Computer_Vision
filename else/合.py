import cv2
import math
import numpy as np

def Img_Outline(input_dir):
    original_img = cv2.imread(input_dir)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)                     # 高斯模糊去噪（设定卷积核大小影响效果）
    _, RedThresh = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)  # 设定阈值165（阈值影响开闭运算效果）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))          # 定义矩形结构元素
    closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)       # 闭运算（链接块）
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)           # 开运算（去噪点）
    return original_img, gray_img, RedThresh, closed, opened


def findContours_img(original_img, opened):
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]   # 计算最大轮廓的旋转包围盒
    rect = cv2.minAreaRect(c)                                    # 获取包围盒（中心点，宽高，旋转角度）
    box = np.int0(cv2.boxPoints(rect))                           # box
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)

    print("box[0]:", box[0])
    print("box[1]:", box[1])
    print("box[2]:", box[2])
    print("box[3]:", box[3])
    return box,draw_img

def Perspective_transform(box,original_img):
    # 获取画框宽高(x=orignal_W,y=orignal_H)
    orignal_W = math.ceil(np.sqrt((box[3][1] - box[2][1])**2 + (box[3][0] - box[2][0])**2))
    orignal_H= math.ceil(np.sqrt((box[3][1] - box[0][1])**2 + (box[3][0] - box[0][0])**2))

    # 原图中的四个顶点,与变换矩阵
    pts1 = np.float32([box[0], box[1], box[2], box[3]])
    print(pts1)
    pts2 = np.float32([[0, 0], [int(orignal_W+1), 0], [int(orignal_W+1),int(orignal_H+1)],[0, int(orignal_H+1)]])

    # 生成透视变换矩阵；进行透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result_img = cv2.warpPerspective(original_img, M, (int(orignal_W+3),int(orignal_H+1)))

    return result_img


def adaptiveThresh(I, winSize, ratio=0.06):
    # 第一步:对图像矩阵进行均值平滑
    I_mean = cv2.boxFilter(I, cv2.CV_32FC1, winSize)

    # 第二步:原图像矩阵与平滑结果做差
    out = I - (1.0 - ratio) * I_mean

    # 第三步:当差值大于或等于0时，输出值为255；反之，输出值为0
    out[out >= 0] = 255
    out[out < 0] = 0
    out = out.astype(np.uint8)
    return out


if __name__=="__main__":
    input_dir = "test8.jpg"
    original_img, gray_img, RedThresh, closed, opened = Img_Outline(input_dir)
    box, draw_img = findContours_img(original_img,opened)
    result_img = Perspective_transform(box,original_img)
    final_img = adaptiveThresh(result_img, (5, 5))
    # final_img = cv2.resize(final_img, (700, 600))
    cv2.imshow("original", original_img)
    cv2.imshow("gray", gray_img)
    # cv2.imshow("closed", closed)
    # cv2.imshow("opened", opened)
    cv2.imwrite("test8_closed.jpg", closed)
    # cv2.imshow("draw_img", draw_img)
    cv2.imwrite("test8_gray.jpg", draw_img)
    # cv2.imshow("result_img", final_img)
    cv2.imwrite("test8_final.jpg", final_img)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
