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

#read images
from skimage.io import imread, imshow
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

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




#%% #load images to dict for training

# Define the path to the directory containing the images
train_csv = r'image_path.csv'

def load_images(path):
    img = imread(path)
    img = gray2rgb(img) #convert to rgb for transfer learning
    return np.array(np.float16(img))
 

with open(train_csv,"r") as f:
    data_dict = {'image': [], 'label': []}
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        image_path = row[0]
        data_dict['image'].append(load_images(image_path))     
        data_dict['label'].append(np.uint8(row[1]))



#%% Load torch dataset


# Define the transforms to apply to the images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
    ])


# Define the fraction of data to use for validation
val_fraction = 0.2

# Split the data into training and validation sets
train_size = int(len(data_dict['image']) * (1 - val_fraction))
val_size = len(data_dict) - train_size
train_data, val_data = data.random_split(data_dict, [train_size, val_size])


# Create the dataloaders
train_dataset = data.TensorDataset(transform(train_data['image']), torch.tensor(train_data['label']))
train_dataloader = data.DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=4)

val_dataset = data.TensorDataset(transform(val_data['image']), torch.tensor(val_data['label']))
val_dataloader = data.DataLoader(val_dataset, batch_size=3, shuffle=False, num_workers=4)



#%% Run models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_classes=2
model = torchvision.models.resnet50(pretrained=True)


# %% build neural net

model.cuda()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    # Train the model for one epoch
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    model.train()
    running_loss = 0.0
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(inputs)
        labels = labels.int64()
        loss = criterion(predictions, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)

    # Evaluate the model on the validation set
    model.eval()
    running_corrects = 0
    for inputs, labels in val_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        labels = labels.int64()

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    epoch_acc = running_corrects.double() / len(val_dataset)

    # Print the loss and accuracy for this epoch
    print('Epoch [{}/{}], Loss: {:.4f}, Val Acc: {:.4f}'.format(epoch+1, num_epochs, epoch_loss, epoch_acc))