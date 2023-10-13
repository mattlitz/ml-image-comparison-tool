########################
#
# ML Image Comparison Tool
#
# author: @mattlitz
#%% 
import os
import numpy as np
import pandas as pd

#read images
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

import mlflow
mlflow.autolog()


#%% 
#pytorch neural nets
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms



import lightning as L
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint



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


#%% Load torch dataset



#%% Run models

img = imread('images/train/C0001_0001.png')
imshow(img)


# %%
