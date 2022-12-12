import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
#from tensorflow.keras.models import load_model 
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from keras.datasets import mnist



"""
model = load_model('filename')

def prediciton(image, model):
  img = cv2.resixe(image, (28, 28))
  img = img / 255
  img = img.reshape(1, 28, 28, 1)
  predict = model.predict(img)
  prob = np.amax(predict)
  class_index = model.predict_classes(img)
  result = class_index[0]
  if prob < 0.75:
    result = 0
    prob = 0
  return result, prob
"""
# then add the text and the prediction in the for loop

# Folder path variable to add the card
folder_path = r'D:/GitHub/recognise_uno_cards/images/'

card = ''
card_colors = ['b', 'g', 'y', 'r']
# yellow is the problematic colour since it doesn't show up in B/W card | it also has a lot of shadows in numbers

for c in range(10):
  card = card_colors[1] + str(c)    # is used to combine the number and the color letter in a string
  img_colour = cv2.imread(folder_path + card + '.jpg')  # open the saved image in colour
  img = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)   # convert to B/W
  estimatedThreshold, thresholdImage=cv2.threshold(img,160,255,cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(thresholdImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   # extract contours on binarised image, not on canny

  #img_sm = cv2.blur(img, (1, 1))         # smoothing
  #thr_value, img_th = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)   # binarisation
  #kernel = np.ones((5, 5), np.uint8)
  #img_close = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)      # morphology correction
  #img_canny = cv2.Canny(img_close, 50, 100)                          # edge detection

  #data = []

  for i, c in enumerate(contours):         # loop through all the found contours
      if hierarchy[0,i,3] != -1 and hierarchy[0,i,3] != 0:
        print(i, ':', hierarchy[0, i])          # display contour hierarchy
        print('length: ', len(c))               # display numbr of points in contour c
        perimeter = cv2.arcLength(c, True)     # perimeter of contour c (curved length)
        print('perimeter: ', perimeter)               
        epsilon = 0.02*perimeter    # parameter of polygon approximation: smaller values provide more vertices
        vertex_approx = len(cv2.approxPolyDP(c, epsilon, True))     # approximate with polygon
        print('approx corners: ', vertex_approx, '\n')                    # number of vertices
        area = cv2.contourArea(c)
        print('area: ', area, '\n')
        if len(contours[i]) >= 5:
          ellipse = cv2.fitEllipse(c)
        (center, axes, orientation) = ellipse
        majoraxis_length = max(axes)
        minoraxis_length = min(axes)
        (xc,yc),(d1,d2),angle = ellipse
        axes_ratio = round(d1/d2, 3)
        print('axes ratio: ', axes_ratio, '\n')
        cv2.drawContours(img_colour, [c], 0, (0, 255, 0), 2)   # paint contour c
        cv2.putText(img_colour, str(i), (c[0, 0, 0]+20, c[0, 0, 1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))    # identify contour c
        #[x,y,w,h] = cv2.boundingRect(c)
      # cv2.rectangle(img_colour, (x,y), (x+w,y+h), (255, 0, 0), 2)
        cv2.ellipse(img_colour, ellipse, (255, 0, 0), 2)


      # add an if statement to only add the contour that is biggest between the number in center and on the sides

      # I can crop the image by the biggest contour and then add a feature to see if it has any children
      
      #sample = [] #add the features here so that I can use data to extract all the features to do machine learning 
      #data.append(sample)
  X = numbers.data
  y = numbers.target
  corners = vertex_approx
  shape_complexity = area/perimeter
  relative_length = minoraxis_length/majoraxis_length
  features = [corners, shape_complexity, axes_ratio, relative_length]

  filename = card + '.jpg'
  colour_label = filename[0]
  number_label = int(filename[1])

  np.concatenate(X, features)
  np.concatenate(y, number_label)


  cv2.namedWindow('picture', cv2.WINDOW_NORMAL)
  cv2.imshow('picture',img_colour)
  cv2.imshow('Image', thresholdImage)
  key = cv2.waitKey(0)
  cv2.destroyAllWindows()

  #print(data)
  # label = [ 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9] #add each number times 4 for each colour
  # then add labels to each cards within data
  # can use filename[0] to add label
