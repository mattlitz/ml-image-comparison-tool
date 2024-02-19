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
from transformers import ViTFeatureExtractor, ViTModel

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




#%% Load torch dataset

class trainDataset(torch.utils.data.Dataset):
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





# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(224 * 224, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)  # Latent space
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 224 * 224),
            nn.Sigmoid()  # To get the output in the range [0, 1]
        )
        self.classifier = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classified = self.classifier(encoded)
        return decoded, classified






#%% Split training data into 80/20 train/val




def train():

    # Train the model
    
    #for epoch in range(num_epochs):
    for batch, (images, labels) in enumerate(train_dataloader):
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



def test():

    # Evaluate the model on the validation set
    model.eval()

    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        labels = labels.int64()

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    epoch_acc = running_corrects.double() / len(val_dataset)

    # Print the loss and accuracy for this epoch
    print('Epoch [{}/{}], Loss: {:.4f}, Val Acc: {:.4f}'.format(epoch+1, num_epochs, epoch_loss, epoch_acc))









# %% Run training




if __name__ == "__main__":
    


    # Define the loss function and optimizer
    model = Autoencoder()    
    model.cuda()
    
    criterion1 = nn.MSELoss()  # For reconstruction
    criterion2 = nn.BCELoss()  # For classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)



    # Define the transforms to apply to the images
    transform = {
        "train": v2.Compose([
            v2.RandomHorizontalCrop(224),
            v2.ToDtype(),
            v2.ToTensor()
        ]),
        "test": v2.Compose([
            v2.RandomHorizontalCrop(224),
            v2.ToDtype(),
            v2.ToTensor()
        ])
    }



    train_dataset = trainDataset(data_dict, transform=transform)
    train_data, val_data = train_test_split(train_dataset, test_size=0.2, random_state=42)

    # Create the dataloaders
    train_dataloader = data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
    test_dataloader = data.DataLoader(val_data, batch_size=4, shuffle=False, num_workers=4)

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])

    num_epochs = 10
    for epoch in range(num_epochs):
        train_acc, train_loss = train()
        test_acc, test_loss = test()
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)