import cv2
import numpy as np

kernel = np.ones((5, 5), np.uint8)
vc = cv2.VideoCapture(0)

while vc.isOpened():
    rval, frame = vc.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thr_value, img_thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
    img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel, iterations = 2)
    img_canny = cv2.Canny(img_close, 100, 200)
    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours,-1,(0,255,0),1)
    cv2.imshow('stream', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
        
cv2.destroyWindow('stream')
vc.release()