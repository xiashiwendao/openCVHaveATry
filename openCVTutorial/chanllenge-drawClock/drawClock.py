import cv2
import numpy as np

if __name__ == "__main__":
    drawing = np.zeros((300, 300), dtype=np.uint8)
    # cv2.imshow('drawing_empty', drawing)
    # cv2.waitKey(0)
    cv2.circle(drawing, (150, 150), 50, 255, lineType=cv2.LINE_AA)
    # cv2.imshow('circle', drawing)
    # cv2.waitKey(0)
    cv2.line(drawing, (150, 100), (150, 120), 255, lineType=cv2.LINE_AA)
    cv2.imshow('circle', drawing)
    cv2.waitKey(0)