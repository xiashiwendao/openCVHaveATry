import cv2
cap = cv2.VideoCapture(1)
cap.isOpened()
while 1:
    ret, frame = cap.read()
    cv2.imshow("capture", frame)
    if cv2.waitKey(100) and 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()