import cv2
import numpy as np
import os
import pickle
import csv
import pandas as pd

kernel = np.ones((5, 5), np.uint8)
vc = cv2.VideoCapture(0)
file = pickle.load(open('digits_classifier.p', "rb"))    


while vc.isOpened(): 
    rval, frame = vc.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thr_value, img_thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
    img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel, iterations = 2)
    img_canny = cv2.Canny(img_close, 100, 200)
    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):  
        perimeter = cv2.arcLength(c, True)     # perimeter of contour c (curved length)            
        epsilon = 0.02*perimeter    # parameter of polygon approximation: smaller values provide more vertices
        vertex_approx = len(cv2.approxPolyDP(c, epsilon, True))     # approximate with polygon
        area = cv2.contourArea(c)
        if len(contours[i]) >= 5:
            ellipse = cv2.fitEllipse(c)
        (center, axes, orientation) = ellipse
        majoraxis_length = max(axes)
        minoraxis_length = min(axes)
        (xc,yc),(d1,d2),angle = ellipse
        axes_ratio = round(d1/d2, 3)

        #Feature Extraction
        corners = vertex_approx
        #shape_complexity = area/perimeter
        relative_length = minoraxis_length/majoraxis_length
        features = [area, perimeter, axes_ratio, relative_length]
        features_array = (np.array(features)).reshape(1, -1)

        y_predict = file.predict(features_array)
        result = str(y_predict)
        cv2.putText(frame, result, (c[0, 0, 0]+20, c[0, 0, 1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    cv2.drawContours(frame, contours,-1,(0,255,0),1)
    cv2.imshow('stream', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyWindow('stream')
vc.release()