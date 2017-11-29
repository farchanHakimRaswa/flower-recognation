# import the necessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import feature as ft
import numpy as np
import mahotas
import cv2
import os
import h5py
import random
import json
import seaborn as sns


with open('../config/conf.json') as f:    
	config = json.load(f)

train_path 	= config["dataset"]
feature_path 	= config["features_path"]
labels_path 	= config["labels_path"]

# fixed-sizes for image
fixed_size = tuple((500, 500))

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels = []

i, j = 0, 0
k = 0


#time
# loop over the training data sub-folders
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name
    k = 1
    # loop over the images in each sub-folder
    for x in range(1,350):
	a=random.choice(os.listdir(dir))
	file2 = dir+'/'+a
	print file2
        # read the image and resize it to a fixed-size
        image = cv2.imread(file2)
        image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        fv_hu_moments = ft.fd_hu_moments(image)
        fv_haralick   = ft.fd_haralick(image)
        fv_histogram  = ft.fd_histogram(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)
	print global_feature
        #print global_feature
        i += 1
        k += 1
    print "[STATUS] processed folder: {}".format(current_label)
    j += 1

print "[STATUS] completed Feature Extraction..."


#time
# get the overall feature vector size
print "[STATUS] feature vector size {}".format(np.array(global_features).shape)

# get the overall training label size
print "[STATUS] training Labels {}".format(np.array(labels).shape)

# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print "[STATUS] training labels encoded..."

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print "[STATUS] feature vector normalized..."

print "[STATUS] target labels: {}".format(target)
print "[STATUS] target labels shape: {}".format(target.shape)

# save the feature vector using HDF5
h5f_data = h5py.File(feature_path, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print "[STATUS] end of training.."

