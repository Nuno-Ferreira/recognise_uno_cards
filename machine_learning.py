import cv2
import numpy as np
import os
import pickle
import csv
import pandas as pd
import matplotlib.pyplot as plt
#from tensorflow.keras.models import load_model 
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from keras.datasets import mnist

# with open("dataset.csv") as csv_file:
#     csv_reader = csv.reader(csv_file)

data = pd.read_csv('dataset.csv')

x = data.drop(data.columns[[-1]], axis=1)
y = data.drop(data.columns[[0,1,2,3,4]], axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=42)
classifier = MLPClassifier()
classifier.fit(x_train,y_train)
pickle.dump(classifier, open("digits_classifier.p", "wb"))

clf = pickle.load(open("iris_classifier.p", "rb"))    # load the model to test it (usually in a different script)
score = clf.score(x_test, y_test)
print(score)
