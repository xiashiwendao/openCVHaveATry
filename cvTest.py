import cv2
file_path = r'C:\Users\python_gay\MySpace\Code_Space\openCVHaveATry\images\car_no\train_no\0.bmp'
img = cv2.imread(file_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow('img_gary', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np
d = np.zeros((3,5))
print(d[3,:])