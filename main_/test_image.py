# import the necessary packages
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.externals import joblib
import feature as ft
import matplotlib.pyplot as plt
import mahotas
import pickle
import json
import h5py
import numpy as np
import os
import glob
import cv2
import random
import sys


with open('../config/conf.json') as f:    
	config = json.load(f)

test_path 	= config['test_path']
classifier_path = config['classifier_path']

fixed_size = tuple((500, 500))

# load the model from disk
clf = pickle.load(open(classifier_path, 'rb'))

file2 = "/home/achan/Documents/paper/Projekdwiki/ls/main_"+'/'+str(sys.argv[1]);
print file2
image = cv2.imread(file2)
   
# resize the image
image = cv2.resize(image, fixed_size)

####################################
# Global Feature extraction
####################################
fv_hu_moments = ft.fd_hu_moments(image)
fv_haralick   = ft.fd_haralick(image)
fv_histogram  = ft.fd_histogram(image)
fv_hog	  = ft.fd_hog_descriptor(image)
###################################
# Concatenate global features
###################################
global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

# predict label of test image
prediction = clf.predict(global_feature.reshape(1,-1))[0]

if prediction == 0:
	print "BUNGANYA ADALAH LOTUS"
elif prediction == 1:
	print "BUNGANYA ADALAH MAWAR"
elif prediction == 2:
	print "BUNGANYA ADALAH MELATI"
elif prediction == 3:
	print "BUNGANYA ADALAH SUNFLOWER"
elif prediction == 4:
	print "BUNGANYA ADALAH TULIP"
