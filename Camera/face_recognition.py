import cv2
import os, sys
import numpy as np

def generate():
    face_cascade =  cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(1)
    count = 0
    while (True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            f = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            cv2.imwrite('images\\face_capture\\%s.pgm' % str(count), f)
            print(os.path.exists('images\\face_capture\\'))
            count += 1
        cv2.imshow("camera", frame)
        if cv2.waitKey(int(1000 / 12)) & 0xff == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

def read_images(path, sz=None):
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            if not os.path.isdir(subject_path):
                continue
            print('++++++++++++ subdirname:', subdirname, '+++++++++++++++++++')
            for filename in os.listdir(subject_path):
                try:
                    if (filename == ".directory"):
                        continue
                    filepath = os.path.join(subject_path, filename)
                    print('========== filepath: ', filepath, ' =============')
                    if(not os.path.exists(filepath)):
                        print('++++++++ My God, file not exists!')
                    
                    # im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    # im = cv2.imread(filepath,cv2.IMREAD_UNCHANGED)
                    img = Image.open(filepath)
                    img2cv = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
                    # cv2.imshow('pic',im)
                    # cv2.waitKey()
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(img2cv, (200, 200))
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                # except IOError, (errno, strerror):
                #     print("I/O error({0}): {1}".format(errno, strerror))
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            c = c+1
    
    return [X,y]

def face_rec():
    names = ['liuyifei', 'Lorry', 'Yangmi']
    # if len(sys.argv) < 2:
    #     print("USAGE: facerec_demo.py </path/to/images> [</path/to/store/images/at>]")
    #     sys.exit()
    [X,y] =  read_images('images\\face_capture', True)
    y = np.asarray(y, dtype=np.int32)
    # if len(sys.argv) == 3:
    #     out_dir = sys.argv[2]
    # model = cv2.face.EigenFaceRecognizer_create()
    model = cv2.face.FisherFaceRecognizer_create()
    model.train(np.asarray(X), np.asarray(y))
    camera = cv2.VideoCapture(1)
    face_cascade =cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    while (True):
        read, img = camera.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[x:x+w, y:y+h]
            try:
                roi = cv2.resize(roi, (200, 200),interpolation=cv2.INTER_LINEAR)
                params = model.predict(roi)
                print("Label: %s, Confidence: %.2f" % (params[0], params[1]))
                cv2.putText(img, names[params[0]], (x, y - 20),cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            except:
                continue

        cv2.imshow("camera", img)
        if cv2.waitKey(int(1000 / 12)) & 0xff == ord("1"):
            break
        
    cv2.destroyAllWindows()

def face_rec_by_img(img_path):
    names = []#['liuyifei', 'Lorry', 'Yangmi']
    for i in range(16):
        index = ''
        if i<10:
            index='subject0'+str(i)
        else:
            index='subject'+str(i)

        names.append(index)
    
    print(names)
    # if len(sys.argv) < 2:
    #     print("USAGE: facerec_demo.py </path/to/images> [</path/to/store/images/at>]")
    #     sys.exit()
    [X,y] =  read_images('C:\\MySpace\\research\\machinelearning\\OpenCV\\Yale_face\\yalefaces\\', True)
    y = np.asarray(y, dtype=np.int32)
    # if len(sys.argv) == 3:
    #     out_dir = sys.argv[2]
    # model = cv2.face.EigenFaceRecognizer_create()
    # model = cv2.face.FisherFaceRecognizer_create()
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(X), np.asarray(y))
    face_cascade =cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    
    img = cv2.imread(img_path)
    # cv2.imshow("ori", img)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi = gray[x:x+w, y:y+h]
        try:
            roi = cv2.resize(roi, (200, 200),interpolation=cv2.INTER_LINEAR)
            params = model.predict(roi)
            print("Label: %s, Confidence: %.2f" % (params[0]+1, params[1]))
            print('+++++ params: ', params, '++++++++')
            cv2.putText(img, names[params[0]], (x, y - 20),cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            
            cv2.imshow(img)
        except:
            continue

    if cv2.waitKey(int(1000 / 12)) & 0xff == ord("1"):
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    # face_rec()
    # img_path = 'images\\face_test\\zhang2.jpg'
    # img_path = 'images\\face_capture\\liuyifei\\1.jpg'
    img_path= 'images/face_capture/Yangmi/1.jpg'
    # img_path= 'C:\\MySpace\\research\\machinelearning\\OpenCV\\Yale_face\\yalefaces_validation\\subject01.wink.gif'
    face_rec_by_img(img_path)