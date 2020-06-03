import cv2
import numpy as np

# 1.加载图片，转为二值图
img = cv2.imread(r'c:/Users/python_gay/MySpace/Code_Space/openCVHaveATry/opencvTutorial/17.houghlines/shapes.jpg')
drawing = np.zeros(img.shape[:], dtype=np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

def draw_line():
    # 2.霍夫直线变换
    lines = cv2.HoughLines(edges, 0.8, np.pi / 180, 90)
    # 3.将检测的线画出来（注意是极坐标噢）
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(drawing, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.imshow('hough lines', np.hstack((img, drawing)))
    cv2.waitKey(0)
# 2. 统计概率霍夫线变换
def draw_lines_prob():
    drawing = np.zeros(img.shape[:], dtype=np.uint8)
    lines = cv2.HoughLinesP(edges, 0.8, np.pi / 180, 90, minLineLength=50, maxLineGap=10)
    # 将检测的线画出来
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # line_aa代表抗锯齿
        cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)

    cv2.imshow('probabilistic hough lines', np.hstack((img, drawing)))
    cv2.waitKey(0)

def draw_circle():
    drawing = np.zeros(img.shape[:], dtype=np.uint8)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param2=30)
    # circles = np.int0(np.around(circles))
    # 将检测的圆画出来
    for i in circles[0, :]:
        cv2.circle(drawing, (i[0], i[1]), i[2], (0, 255, 0), 2, lineType=cv2.LINE_AA)  # 画出外圆
        cv2.circle(drawing, (i[0], i[1]), 2, (0, 0, 255), 3, lineType=cv2.LINE_AA)  # 画出圆心

    cv2.imshow('circles', np.hstack((img, drawing)))
    cv2.waitKey(0)


if __name__ == "__main__":
    # draw_lines_prob()
    draw_circle()
