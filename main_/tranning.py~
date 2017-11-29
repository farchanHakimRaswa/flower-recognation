# import the necessary packages
import h5py
import numpy as np
import os
import glob
import cv2
import itertools
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import pickle
import json
import random
import mahotas

# filter all the warnings
import warnings
warnings.filterwarnings('ignore')


with open('../config/conf.json') as f:    
	config = json.load(f)

feature_path 	= config['features_path']
labels_path 	= config['labels_path']
classifier_path = config['classifier_path']


# bins for histogram
bins = 8
# fixed-sizes for image
fixed_size = tuple((500, 500))


# train_test_split size
test_size = 0.10

# seed for reproducing same results
seed = 9

# variables to hold the results and names
results = []
names = []
scoring = "accuracy"


# import the feature vector and trained labels
h5f_data = h5py.File(feature_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print "[STATUS] features shape: {}".format(global_features.shape)
print "[STATUS] labels shape: {}".format(global_labels.shape)

print "[STATUS] training started..."


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#PRE PROCESSING
#Scaling the values
data = scale(np.array(global_features))

#SELEKSI FITUR
pca = PCA(n_components=10)
pca.fit(data)
d=pca.fit_transform(data)
print(pca.components_)

# split the training and testing data
X_train, X_test, Y_train, Y_test  = train_test_split(np.array(global_features), np.array(global_labels),                          							      test_size=0.3, random_state=seed)
print "[STATUS] splitted train and test data..."
print "Train data  : {}".format(X_train.shape)
print "Test data   : {}".format(X_test.shape)
print "Train labels: {}".format(Y_train.shape)
print "Test labels : {}".format(Y_test.shape)
print "[STATUS] splitted train and test data..."
print "size dimension data sebelum	: ",data.shape
print "size dimension data sebelum   	: ", d.shape


#model = SVC(C=1000.0, kernel="rbf", gamma=10)
model = RandomForestClassifier(n_estimators=100, random_state=9)
model.fit(X_train, Y_train)

Y_predict = model.predict(X_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, Y_predict)

kelas = ['lotus', 'mawar', 'melati', 'sunflower','tulip'];
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=kelas,
                      title='Confusion matrix,')

akurasi = accuracy_score(Y_test, Y_predict)
presisi = precision_score(Y_test, Y_predict, average='weighted') 
print "AKURASI = ", akurasi*100;
print "PRESISI = ", presisi*100;
print "ERROR = ", 100 - (akurasi*100);

#save data classifier nih
pickle.dump(model, open(classifier_path, 'wb'))

plt.show()



