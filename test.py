#Basic steps for finding the number of corners in a contour (depends on the polygon_constant you choose)
perimeter = cv2.arcLength(contour, True)   # contour length, also potentially useful
polygon_constant = 0.04    # try changing this value to get more or less corners
vertex_approx = len(cv2.approxPolyDP(contour, polygon_constant*perimeter, True))   # number of corners: how can you change this to a [0,1] interval ?

# Basic steps for finding the major axes of the shape described by the contour, their length, orienbtation and shape centre
ellipse = cv2.fitEllipse(contour)    # fit an ellipse on the contour
(center, axes, orientation) = ellipse   # extract the main parameter
majoraxis_length = max(axes)
minoraxis_length = min(axes)

# Binarising an image by colour in HSV space, you need to set the right values for the colour you want to filter in
HSVmin = [0.0, 0.0, 0.0]   # minimum values of chosen range
HSVmax = [180.0, 255.0, 255.0]   # maximum values of chosen range

# putting features together
corners = vertex_approx
relative_length = minor_axis/major_axis
shape_complexity = area/perimeter
features = [corners, relative_length, shape_complexity]

# you can use the filename to identify the labels
filename = 'b0.jpg'
colour_label = filename[0]
number_label = int(filename[1])

# append the feature vector to the feature space (X matrix), and the label to your classes (y array)
# same as it looks in ML_classification notebook:
np.concatenate(X, features) # OR X.append(features)
np.concatenate(y, number_label)

# use pickle to save a trained model
import cv2
import pickle
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.8, random_state=42)
classifier = MLPClassifier()
classifier.fit(X_train, y_train)
pickle.dump(classifier, open("iris_classifier.p\", \"wb"))    # save the model after training, remember the file name
#classifier = []
#score = classifier.score(X_test, y_test)
    
clf = pickle.load(open("iris_classifier.p\", \"rb"))    # load the model to test it (usually in a different script)
score = clf.score(X_test, y_test)
print(score)

