########################
#
# ML Image Comparison Tool
#
# author: @mattlitz

import os
import numpy as np
import pandas as pd

from sklearnex import patch_sklearn
patch_sklearn()

#scikit-learn methods
from sklearn import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#pytorch neural nets
import torch


import mlflow
mlflow.autolog()


#%% 

#use dataframe to store images for training




#%% 

