########################
#
# ML Image Comparison Tool
#
# author: @mattlitz
#%% 
import os
import numpy as np
import pandas as pd

from sklearnex import patch_sklearn
patch_sklearn()

#read images
from skimage.io import imread, imshow
import matplotlib.pyplot as plt

#scikit-learn methods
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression
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


#%% Tune hyperparameters




#%% Run models

img = imread('images/train/C0001_0001.png')
imshow(img)


# %%
