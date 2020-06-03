import cv2
import numpy as np
import matplotlib.pyplot as plt
img_path = 'opencvTutorial\\img\\hist.jpg'
img_path_2 = 'opencvTutorial\\img\\tsukuba.jpg'
# plt show的图片最好是plt imread的（numpy形式），如果是cv2.imread的需要使用numpy进行转换
def plt_showhist(img):
    img = cv2.imread(img_path, 0)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # 性能：0.025288 s
    hist_plt = plt.imread(img_path)
    img = np.array(img)
    #plt.plot(hist)
    plt.hist(hist_plt.ravel(), 256, [0, 256])
    plt.show()

# 灰度图经过均衡直方图之后，会变得更加黑白分明，从图像分布来看也更加均衡
def equalize_pic():
    img = cv2.imread(img_path, 0)
    equ_img = cv2.equalizeHist(img)
    # cv2.imshow('equalization', np.hstack((img, equ_img)))
    # cv2.waitKey(0)
    equ_img = np.array(equ_img)
    plt.hist(equ_img, 256, [0, 256])
    plt.show()

import sys, os
def equalize_pic_2():
    img = cv2.imread(img_path_2, cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread(img_path_2)
    equ_img = cv2.equalizeHist(img)
    cv2.imshow('equliaze', np.hstack((img, equ_img)))
    cv2.waitKey(0)

def adapterHist():
    img = cv2.imread(img_path_2, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(18, 18))
    cl2 = clahe.apply(img)

    cv2.imshow('equliaze', np.hstack((cl1, cl2)))
    
    cv2.waitKey(0)
# 计算指定区域进行直方图
def homework():
    img = cv2.imread(img_path, 0)
    hist = cv2.calcHist([img], [0], [200, 200], [256], [0, 256])  # 性能：0.025288 s
 
if __name__ == "__main__":
    # equalize_pic_2()
    adapterHist()
