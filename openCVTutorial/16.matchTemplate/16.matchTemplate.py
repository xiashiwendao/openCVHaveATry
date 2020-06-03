import cv2
import numpy as np
from matplotlib import pyplot as plt
import os, sys

img_folder = r'C:\Users\python_gay\MySpace\Code_Space\openCVHaveATry\opencvTutorial\16.matchTemplate'
img = cv2.imread(os.path.join(img_folder, 'lena_bigger.jpg'), 0)
template = cv2.imread(os.path.join(img_folder,'face.jpg'), 0)
h, w = template.shape[:2]  # rows->h, cols->w
# 相关系数匹配方法：cv2.TM_CCOEFF
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print('min_val: ', min_val, '; max_val:', max_val, '; min_loc: ', min_loc, '; max_loc: ', max_loc)
left_top = max_loc  # 左上角
right_bottom = (left_top[0] + w, left_top[1] + h)  # 右下角
img_rec = cv2.rectangle(img, left_top, right_bottom, 255, 2)  # 画出矩形位置
cv2.imshow("img", img_rec)
cv2.waitKey(0)