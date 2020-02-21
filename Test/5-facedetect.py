import cv2
jpgfilename = 'images\\oneman2.jpg'
# xmlfile = 'cascades\\haarcascade_frontalface_default.xml'
# xmlfile='C:\\Users\\Lorry\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
xmlfile = 'cascades\\haarcascade_lefteye_2splits.xml'
import sys, os
def detect(filename):
    face_cascade = cv2.CascadeClassifier(xmlfile)
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.namedWindow('Vikings Detected!!')
    cv2.imshow('Vikings Detected!!', img)
    cv2.imwrite('vikings.jpg', img)
    cv2.waitKey(0)

if os.path.exists(jpgfilename):
    print('file exists, next')
    if os.path.exists(xmlfile):
        print('xml file exiss, next')
        detect(jpgfilename)
    else:
        print('NG! xml file not exists!')
else:
    print('NG! jpgfile not exists!')
