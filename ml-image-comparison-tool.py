########################
#
# ML Image Comparison Tool
#
# author: @mattlitz
#%% 
import os
import numpy as np
import pandas as pd
import csv
import generate_textures as gt

from sklearnex import patch_sklearn
patch_sklearn()

#read images
from skimage.io import imread, imshow
import matplotlib.pyplot as plt

#scikit-learn methods
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#basyesian optimization
from skopt import gp_minimize

#res = gp_minimize(f,                  # the function to minimize
#                  [(-2.0, 2.0)],      # the bounds on each dimension of x
#                  acq_func="EI",      # the acquisition function
#                  n_calls=15,         # the number of evaluations of f
#                  n_random_starts=5,  # the number of random initialization points
#                  noise=0.1**2,       # the noise level (optional)
#                  random_state=1234)   # the random seed

#pytorch neural nets
import torch


import mlflow
mlflow.autolog()
#run mlflow ui to start mlflow server and then http://localhost:5000


#%% 

#use dataframe to store images for training
data_dict = {}
train_data = pd.DataFrame()

# Define the path to the directory containing the images
train_path = "images/train/"

for images in os.listdir(train_path):
    img = imread(os.path.join(train_path, images))
    img = np.array(img)
    data_dict[images] = img
    

for key, value in data_dict.items():
    row = pd.Series([key, value])
    train_data = pd.concat([train_data, pd.DataFrame([[key, value]])], ignore_index=True)
train_data = train_data.rename(columns={0: 'filename', 1: 'image'})


#%% #load images to dict for training

# Define the path to the directory containing the images
train_csv = r'image_path.csv'

def load_images(path):
    img = imread(path)
    return np.array(np.uint8(img))
 

with open(train_csv,"r") as f:
    data_dict = {'image': [], 'filename': [], 'label': [], 'entropy': [], 'otsu_segment': [], 'kmeans_segment': []}
    reader = csv.DictReader(f)
    for row in reader:
        image_path = row['path']
        label_name = row['label']
        data_dict['image'].append(load_images(image_path))   
        data_dict['filename'].append(image_path)   
        data_dict['label'].append(label_name) 
        data_dict['entropy'].append(gt.entropy_img(load_images(image_path),10))
        data_dict['otsu_segment'].append(gt.otsu_segment(load_images(image_path)))
        data_dict['kmeans_segment'].append(gt.kmeans_segment(load_images(image_path),5)) 



#%% Tune hyperparameters






#%% Classify image with GPC

images = data_dict['image']
labels = data_dict['label']

images = [img.flatten() for img in images]
scaler = StandardScaler()
images = scaler.fit_transform(images)

# Split the data into training and testing sets
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the classifier
gpc = GaussianProcessClassifier(random_state=0)

# Train the classifier
gpc.fit(images_train, labels_train)

# Predict the labels of the test set
labels_pred = gpc.predict(images_test)


#%% Regress image with GPR

# Assume 'data_dict' is your dictionary containing two image keys and labels
# 'image1' and 'image2' are lists of images and 'label' is a list of labels
images = data_dict['image']
entropy = data_dict['entropy']
labels = data_dict['label']


# Flatten and standardize images
images = [img.flatten() for img in images]
entropy = [img.flatten() for img in entropy]
scaler = StandardScaler()
images = scaler.fit_transform(images)
entropy = scaler.fit_transform(entropy)

# Combine the two image datasets
textures = np.hstack((images, entropy))

# Split the data into training and testing sets
images_train, images_test, labels_train, labels_test = train_test_split(textures, labels, test_size=0.2, random_state=42)

# Define the regressor
gpr = GaussianProcessRegressor(random_state=0)

# Train the regressor
gpr.fit(images_train, labels_train)

# Predict the labels of the test set
labels_pred = gpr.predict(images_test)


# %% Calculate metrics
