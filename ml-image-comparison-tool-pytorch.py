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

#import mlflow
#mlflow.autolog()


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


writer = SummaryWriter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#%% #load images to dict for training

# Define the path to the directory containing the images
train_csv = r'image_path.csv'

def load_images(path):
    img = imread(path)
    img = gray2rgb(img) #convert to rgb for transfer learning
    return np.array(np.uint8(img))
 

with open(train_csv,"r") as f:
    data_dict = {'image': [], 'label': []}
    reader = csv.DictReader(f)
    for row in reader:
        image_path = row['path']
        data_dict['image'].append(load_images(image_path))     
        data_dict['label'].append(np.uint8(row['label']))



#%% Load dict into torch dataset

class trainDataset(torch.utils.data.Dataset):
    """Some Information about MyDataset"""
    def __init__(self, data_dict, transform=None):
        super(trainDataset, self).__init__()
        self.data_dict = data_dict
        self.transform = transform

    def __getitem__(self, index):

        image = self.data_dict['image'][index]
        label = torch.tensor(self.data_dict['label'][index])

        if self.transform:
            image = self.transform(image)

        return image, label 

    def __len__(self):
        return len(self.data_dict['image'])


# Define the transforms to apply to the images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
    ])



#%% Split training data into 80/20 train/val

train_dataset = trainDataset(data_dict, transform=transform)
train_data, val_data = train_test_split(train_dataset, test_size=0.2, random_state=42)

# Create the dataloaders
train_dataloader = data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
val_dataloader = data.DataLoader(val_data, batch_size=4, shuffle=False, num_workers=4)


#%% Run models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_classes=2
model = torchvision.models.resnet50(pretrained=True)


# %% build neural net

model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the model
num_epochs = 10
#for epoch in range(num_epochs):
for batch, (images, labels) in enumerate(train_dataloader,1):
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
    epoch_acc = running_corrects.double() / len(val_data)

    # Print the loss and accuracy for this epoch
    print('Epoch [{}/{}], Loss: {:.4f}, Val Acc: {:.4f}'.format(num_epochs+1, num_epochs, epoch_loss, epoch_acc))
# %%
