import cv2
import numpy as np

def show_img(img, window_name='window_img'):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
img = cv2.imread('handwrite.jpg')
show_img(img, 'handwrite_show')
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
show_img(thresh, 'thresh')
# contours, hierarchy = cv2.findContours(thresh, 3, 2)
