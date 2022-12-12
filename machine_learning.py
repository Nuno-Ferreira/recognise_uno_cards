import cv2
import numpy as np
import os
import pickle
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



data = pd.read_csv('dataset.csv')


X = data.drop(data.columns[[4,5]], axis=1)
y = data.drop(data.columns[[0,1,2,3,4]], axis=1)
X = np.array(X)
y = np.array(y)


# KNeighborsClassifier

# clf = KNeighborsClassifier(3)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8)
# clf.fit(X_train, y_train)        # train the classifier with the available data
# y_predict = clf.predict(X_test)  # test the classifier on the new data
# y_predict = y_predict.reshape(len(y_test),1)

# score = sum(y_predict == y_test)/len(y_test)  # compare the predicted classes with actual ones and compute the muber of correct guesses
# print(score*100, '%')

# for i in range(len(y_predict)):
#     print(y_predict[i], y_test[i])

# MLPClassifier

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)
classifier = MLPClassifier()
classifier.fit(X_train,y_train)
pickle.dump(classifier, open("digits_classifier.p", "wb"))

clf = pickle.load(open("digits_classifier.p", "rb"))    # load the model to test it (usually in a different script)
score = clf.score(X_test, y_test)
print(score*100)
