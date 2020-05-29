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

def equalize_pic_2():
    img = cv2.imread(img_path_2)
    equ_img = cv2.equalizeHist(img)
    cv2.imshow('equliaze', np.hstack((img, equ_img)))
    cv2.waitKey(0)


if __name__ == "__main__":
    equalize_pic()
