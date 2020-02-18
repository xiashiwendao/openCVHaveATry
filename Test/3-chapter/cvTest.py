import cv2
import numpy
from matplotlib import pyplot as plt



def cannyTest():
    print("hello")
    # # img = plt.imread('images\\dog2.jpg')
    # # plt.imshow(img)
    # # plt.show()

    img = cv2.imread('images\\dog2.jpg')
    # # cv2.imshow(img)
    # # img_canny = cv2.Canny(img, 200, 200)
    # # cv2.imshow(img_canny)

    cv2.imwrite('images\\dog_canny.jpg', cv2.Canny(img, 200, 200))
    cv2.imshow('canny',cv2.imread('images\\dog_canny.jpg'))
    cv2.waitKey()
    cv2.destroyAllWindows()

def contourTest():
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:150, 50:150] = 255
    # 图像进行二值化处理
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 将图像抓换位BGR三通道，之前图像只是黑白一通道的图像（因为后面要花一条有色框，所以要增加通道）
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.drawContours(color, contours, -1, (255,0,0), 2)
    cv2.imshow("contours", color)
    cv2.waitKey()
    cv2.destroyAllWindows()

def complexContourDetect():
    import cv2
    import numpy as np
    img = cv2.pyrDown(cv2.imread("images\\dog_gray.jpg", cv2.IMREAD_UNCHANGED))
    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) ,127, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # find bounding box coordinates
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        # find minimum area
        rect = cv2.minAreaRect(c)
        # calculate coordinates of the minimum area rectangle
        box = cv2.boxPoints(rect)
        # normalize coordinates to integers
        box = np.int0(box)
        # draw contours
        cv2.drawContours(img, [box], 0, (0,0, 255), 3)
        # calculate center and radius of minimum enclosing circle
        (x,y),radius = cv2.minEnclosingCircle(c)
        # cast to integers
        center = (int(x),int(y))
        radius = int(radius)
        # draw the circle
        img = cv2.circle(img,center,radius,(0,255,0),2)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    cv2.imshow("contours", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # cannyTest()
    # contourTest()
    complexContourDetect()
    # img = cv2.imread("images\\dog2.jpg")
    # img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # # cv2.imread("images\\dog2.jpg")
    # cv2.imwrite('imges\\dogs_gray.jpg',img_gray)
    # cv2.imshow('gary_pic', img_gray)
    # cv2.waitKey()
    # cv2.destroyAllWindows()