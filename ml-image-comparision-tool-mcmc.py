########################
#
# ML Image Comparison Tool - Vision Transformer
# 
#
# author: @mattlitz
#%% 
import os
import numpy as np
import pandas as pd
import csv
import h5py

#read images
from skimage.io import imread, imshow
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

#import mlflow
#mlflow.autolog()


#%% 
#pytorch neural nets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import timm
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2

from transformers import AutoImageProcessor, ViTForImageClassification
from transformers import ViTModel, ViTConfig

writer = SummaryWriter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#%% #load images to dict for training

# Define the path to the directory containing the images

class buildData:
    def __init__(self, data_csv):

        self.data_csv = data_csv
        self.labels, self.images = self.load_to_array()

    def load_images(self, path):
        img = imread(path, as_gray=True)
        img = gray2rgb(img) #convert to rgb for transfer learning

        return np.array(np.float32(img)).transpose(2,0,1)

    
    def load_to_array(self):
        img_list =[]
        lbl_list = []

        with open(self.data_csv,"r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_list.append(self.load_images(row['path']))     
                lbl_list.append(np.uint8(row['label']))

            #convert list to numpy array
            labels = np.array(lbl_list).reshape(-1,1)
            images = np.array(img_list)
            
            return labels, images


#%% Load torch dataset

class buildDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        super(buildDataset, self).__init__()

        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).float()

        self.transform = transform

    def __getitem__(self, idx):

        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label 

    def __len__(self):
        return len(self.labels)

#%% Create model class
"""
class ViTBinaryClassifier(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model
        self.classifier = nn.Linear(self.vit.config.hidden_size, 1)

    def forward(self, pixel_values, pixel_mask):
        outputs = self.vit(pixel_values=pixel_values, pixel_mask=pixel_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits
"""
class ViTBinaryClassifier(nn.Module):
    def __init__(self):
        super(ViTBinaryClassifier, self).__init__()
        # Load the pre-trained ViT model
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        # Replace the last layer for binary classification
        self.vit.head = nn.Linear(self.vit.head.in_features, 1)

    def forward(self, x):
        x = self.vit(x)
        return torch.sigmoid(x)



#%% Split training data into 80/20 train/val

def train(dataloader, model, loss_fn, optimizer):
    train_loss = 0
    train_acc = 0
    correct = 0
    total = 0

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        outputs = model(X)
        train_loss = loss_fn(outputs, y)

        # Backpropagation
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            train_loss, current = train_loss.item(), (batch + 1) * len(X)
            print(f"loss: {train_loss:>7f}  [{current:>5d}/{size:>5d}]")

            predictions = (outputs > 0.5).float()
            correct += (predictions == y).sum().item()
            total += y.size(0)
            train_acc = correct / total

    return train_acc, train_loss


def test(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            predictions = (outputs > 0.5).float()
            test_loss += loss_fn(outputs, y).item()
            correct += (predictions == y).sum().item()
    test_loss /= num_batches
    correct /= size
    test_acc = 100*correct
    print(f"Test Error: \n Accuracy: {(test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_acc, test_loss



# %% Run training

if __name__ == "__main__":

    

    # Load the pre-trained model
    #config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
    #model = ViTModel(config)
    model = ViTBinaryClassifier().to(device)

    # Initialize the binary classifier
    #binary_classifier = ViTBinaryClassifier(model).to(device)
    
  
    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)


    # Define the transforms to apply to the images
    transform = {
        "train": v2.Compose([
            #v2.RandomResizedCrop(224),
            v2.Grayscale(num_output_channels=3),
            v2.Resize((224,224)),
            v2.ToDtype(torch.float32)
        ]),
        "test": v2.Compose([
            #v2.RandomResizedCrop(224),
            v2.Grayscale(num_output_channels=3),
            v2.Resize((224,224)),
            v2.ToDtype(torch.float32)
        ])
    }

    #load data into numpy arrays
    train_csv = r'image_path.csv'
    bd = buildData(train_csv)
      
    X_train, X_test, y_train, y_test = train_test_split(bd.images, bd.labels, test_size=0.2, random_state=42)

    train_dataset = buildDataset(X_train, y_train, transform=transform["train"])
    test_dataset = buildDataset(X_test, y_test, transform=transform["test"])

    # Create the dataloaders
    train_dataloader = data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_dataloader = data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
 

    #write image to Tensorboard
    images, labels = next(iter(train_dataloader))
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    #writer.add_graph(binary_classifier, images)


    #Begin train/test loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_acc, train_loss = train(train_dataloader, model, criterion, optimizer)
        test_acc, test_loss = test(test_dataloader, model, criterion, optimizer)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
    print("Done!")

    writer.close()