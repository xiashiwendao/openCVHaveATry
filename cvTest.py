# -*- coding: utf-8 -*-
import cv2 as cv
import os
# img_path = os.path.join('images', 'pic.jpg')
# img = cv.imread(img_path,cv.IMREAD_COLOR)
# # cv.imshow('image',img)
# cv.imwrite(os.path.join('images', 'pic2.jpg'), img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# print('hehe')

img_raw=cv.imread('images/0a943.PNG',cv.IMREAD_COLOR)
ratio = 400
m=ratio*img_raw.shape[0]/img_raw.shape[1]
    
#压缩图像
img=cv.resize(img_raw,(ratio,int(m)),interpolation=cv.INTER_CUBIC)
# cv.imshow('image_raw', img_raw)
# cv.imshow('image', img)
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('image_gray', gray_img)
maxi=float(gray_img.max())
mini=float(gray_img.min())
print('maxi: ',maxi, "; mini: ", mini)
for i in range(gray_img.shape[0]):
    for j in range(gray_img.shape[1]):
        stretch_strength = 255
        gray_img[i,j]=(stretch_strength/(maxi-mini)*gray_img[i,j]-(stretch_strength*mini)/(maxi-mini))
cv.imshow('image_stretch', gray_img)

for i in range(gray_img.shape[0]):
    for j in range(gray_img.shape[1]):
        stretch_strength = 500
        gray_img[i,j]=(stretch_strength/(maxi-mini)*gray_img[i,j]-(stretch_strength*mini)/(maxi-mini))
cv.imshow('image_stretch_500', gray_img)

cv.waitKey(0)
cv.destroyAllWindows()