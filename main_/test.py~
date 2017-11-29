# import the necessary packages
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
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


with open('../config/conf.json') as f:    
	config = json.load(f)

test_path 	= config['dataset']
classifier_path = config['classifier_path']

fixed_size = tuple((500, 500))

# get the training labels
test_labels = os.listdir(test_path)

# sort the training labels
test_labels.sort()
print test_labels
# load the model from disk
clf = pickle.load(open(classifier_path, 'rb'))


for test_name in test_labels:
    # join the training data path and each species training folder
    dir = os.path.join(test_path, test_name)
    # read the image
    a=random.choice(os.listdir(dir))
    file2 = dir+'/'+a
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
    #print "global_feature", global_feature.shape
    # predict label of test image
    prediction = clf.predict(global_feature.reshape(1,-1))[0]

    if prediction == 0:
	print "BUNGANYA ADALAH LOTUS"
        q= "BUNGANYA ADALAH LOTUS"
    elif prediction == 1:
	print "BUNGANYA ADALAH MAWAR"
	q = "BUNGANYA ADALAH MAWAR"
    elif prediction == 2:
	print "BUNGANYA ADALAH MELATI"
	q = "BUNGANYA ADALAH MELATI"
    elif prediction == 3:
	print "BUNGANYA ADALAH SUNFLOWER"
	q = "BUNGANYA ADALAH SUNFLOWER"
    elif prediction == 4:
	print "BUNGANYA ADALAH TULIP"
	q = "BUNGANYA ADALAH TULIP"
    # show predicted label on image
    cv2.putText(image, q, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
