# -*- coding: utf-8 -*-
import cv2 as cv
import os
img_path = os.path.join('images', 'pic.jpg')
img = cv.imread(img_path,cv.IMREAD_COLOR)
# cv.imshow('image',img)
cv.imwrite(os.path.join('images', 'pic2.jpg'), img)
cv.waitKey(0)
cv.destroyAllWindows()