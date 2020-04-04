import cv2
import numpy as np
from matplotlib import pyplot as plt
img = np.zeros((100, 100,3))
img[:,:,0]=255
# cv2.imshow('img', img)py
plt.imshow(img)
plt.show()