import cv2
import numpy

def detect():
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    # 这里如果穿参数0，就是机器背后摄像头，1则是前置摄像头
    camera = cv2.VideoCapture(1)

    while (True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            # 基于识别出来的脸的基础之上

            # 为了减小搜索区域，将范围具体到人脸范围，但是这样相当于截图，之前的
            # (x,y)变成了(0,0)，所以后面在画识别框的时候，需要再加上
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40,40))
            for (ex,ey,ew,eh) in eyes:
                # 这里将坐标切换回全图里面的坐标，ex和ey需要分别添加上x和y
                ex+=x
                ey+=y
                cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        # eyes = eye_cascade.detectMultiScale(gray, 1.03, 5, 0, (40,40))
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        mirror_frame = numpy.fliplr(frame).copy()
        cv2.imshow("camera", mirror_frame)
        if cv2.waitKey(int(1000 / 12)) & 0xff == ord("q"):
            break
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect()