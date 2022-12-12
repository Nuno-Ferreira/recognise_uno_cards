import cv2
import numpy as np
import pickle
import csv
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


# Folder path variable for the card
folder_path = r'D:/GitHub/recognise_uno_cards/images/'

# Card variables to make up the card filename
print('Input it with this format: b7 for Blue 7')
card = input('What card would you like to have a look?:')
#card = ''
#card_colors = ['b', 'g', 'r', 'y']

file = pickle.load(open('digits_classifier.p', "rb"))  

# This function is used to flatten the data list later on due to it being a list of lists
def flatten(list_of_lists):
  if len(list_of_lists) == 0:
    return list_of_lists
  if isinstance(list_of_lists[0], list):
    return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
  return list_of_lists[:1] + flatten(list_of_lists[1:])


# For loop used to loop through all the numbers
#for c in range(10):
#card = card_colors[0] + str(c)    # is used to combine the number and the color letter in a string
img_colour = cv2.imread(folder_path + card + '.jpg')  # open the saved image in colour
img = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)   # convert to B/W
estimatedThreshold, thresholdImage=cv2.threshold(img,160,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresholdImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   # extract contours on binarised image, not on canny

#img_sm = cv2.blur(img, (1, 1))         # smoothing
#thr_value, img_th = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)   # binarisation
#kernel = np.ones((5, 5), np.uint8)
#img_close = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)      # morphology correction
#img_canny = cv2.Canny(img_close, 50, 100)                          # edge detection


data = []
features_list = []

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

      #Feature Extraction
      #corners = vertex_approx
      shape_complexity = area/perimeter
      relative_length = minoraxis_length/majoraxis_length
      features = [area, shape_complexity, axes_ratio, relative_length]
      #features_list.append(features)
      #features_list.sort(reverse=True)
      features_array = (np.array(features)).reshape(1, -1)

      y_predict = file.predict(features_array)
      result = str(y_predict)
      cv2.putText(img_colour, result, (c[0, 0, 0]+20, c[0, 0, 1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))


      cv2.drawContours(img_colour, [c], 0, (0, 255, 0), 2)   # paint contour c
      #cv2.putText(img_colour, str(i), (c[0, 0, 0]+20, c[0, 0, 1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))    # identify contour c
      cv2.ellipse(img_colour, ellipse, (255, 0, 0), 2)


# Data writing here


cv2.namedWindow('picture', cv2.WINDOW_NORMAL)
cv2.imshow('picture',img_colour)
#cv2.imshow('Image', thresholdImage)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

  # Data writing
  # #colour_label = card[0]
  # number_label = int(card[1])

  # # Appending features and labels to create the data needed for dataset
  # data.append(features_list[0])
  # data.append(number_label)

  # # Writing features and labels to dataset
  # with open("dataset.csv", "a", newline='') as csv_file:
  #   wr = csv.writer(csv_file, dialect='excel')
  #   data_flat = flatten(data)
  #   wr.writerow(data_flat)

#-----------------------------------------------------MACHINE LEARNING------------------------------------------------------
"""
data = pd.read_csv('dataset.csv')


X = data.drop(data.columns[[4,5]], axis=1)
y = data.drop(data.columns[[0,1,2,3,4]], axis=1)
X = np.array(X)
y = np.array(y)


# KNeighborsClassifier

clf = KNeighborsClassifier(3)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8)
clf.fit(X_train, y_train)        # train the classifier with the available data
y_predict = clf.predict(X_test)  # test the classifier on the new data
y_predict = y_predict.reshape(len(y_test),1)

score = sum(y_predict == y_test)/len(y_test)  # compare the predicted classes with actual ones and compute the muber of correct guesses
print(score*100, '%')

for i in range(len(y_predict)):
    print(y_predict[i], y_test[i])

# MLPClassifier

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)
# classifier = MLPClassifier()
# classifier.fit(X_train,y_train)
# pickle.dump(classifier, open("digits_classifier.p", "wb"))

# clf = pickle.load(open("iris_classifier.p", "rb"))    # load the model to test it (usually in a different script)
# score = clf.score(X_test, y_test)
# print(score*100)

"""
