########################
#
# ML Image Comparison Tool - Variational Autoencoder
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



#%% 
#pytorch neural nets
import torch
import torch.utils.data as data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

writer = SummaryWriter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#%% #load images to dict for training

# Define the path to the directory containing the images

class buildData:
    def __init__(self, data_csv):

        self.data_csv = data_csv
        self.labels, self.images = self.load_to_array()

    def load_images(self, path):
        img = imread(path, as_gray=True)
        #img = gray2rgb(img) #convert to rgb for transfer learning

        return np.array(np.float32(img))#.transpose(2,0,1)

    
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

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(572*768, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),  # compressed representation
        )

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 572*768)
        )

    def forward(self, x):
        return self.l1(x)

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
#%% Training loop


if __name__ == "__main__":
    
    
    # Define the transforms to apply to the images
    transform = {
        "train": v2.Compose([
            #v2.RandomResizedCrop(224),
            #v2.Grayscale(num_output_channels=3),
            #v2.Resize((224,224)),
            v2.ToDtype(torch.float32)
        ]),
        "test": v2.Compose([
            #v2.RandomResizedCrop(224),
            #v2.Grayscale(num_output_channels=3),
            #v2.Resize((224,224)),
            v2.ToDtype(torch.float32)
        ])
    }

    #load data into numpy arrays
    train_csv = r'image_path.csv'
    bd = buildData(train_csv)
      
    X_train, X_test, y_train, y_test = train_test_split(bd.images, bd.labels, test_size=0.2, random_state=42)

    train_dataset = buildDataset(X_train, y_train, transform=None)
    test_dataset = buildDataset(X_test, y_test, transform=None)

    # Create the dataloaders
    batch_size=4
    train_loader = data.DataLoader(train_dataset)
    test_loader = data.DataLoader(test_dataset)
 

    #write image to Tensorboard
    #images, labels = next(iter(train_dataloader))
    #grid = torchvision.utils.make_grid(images)
    #writer.add_image('images', grid, 0)

    logger = TensorBoardLogger("tb_logs", name="my_model")
    
    # model
    autoencoder = LitAutoEncoder(Encoder(), Decoder())

    # train model
    trainer = L.Trainer(max_epochs=10, logger=logger)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)
    trainer.test(model=autoencoder, dataloaders=test_loader)

