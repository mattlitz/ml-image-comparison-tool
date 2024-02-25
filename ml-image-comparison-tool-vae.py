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

writer = SummaryWriter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#%% #load images to dict for training

# Define the path to the directory containing the images

class buildData:
    def __init__(self, data_csv):

        self.data_csv = data_csv
        self.labels, self.images = self.load_to_array()

    def load_images(self, path):
        img = imread(path, as_gray=True)
        img = np.expand_dims(img, axis=-1)
        #img = gray2rgb(img) #convert to rgb for transfer learning

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

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(572*768, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 572*768)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 572*768))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 572*768), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    
    return train_loss


def test(epoch, batch_size, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(batch_size, 1, 572, 768)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss


# %% Run training

if __name__ == "__main__":
    

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)

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
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
 

    #write image to Tensorboard
    #images, labels = next(iter(train_dataloader))
    #grid = torchvision.utils.make_grid(images)
    #writer.add_image('images', grid, 0)
    
    epochs = 10
    for epoch in range(1, epochs + 1):
        tr_loss = train(epoch, train_loader)
        te_loss = test(epoch, batch_size, test_loader)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 572, 768),
                        'results/sample_' + str(epoch) + '.png')
            writer.add_scalar('Loss/train', tr_loss, epoch)
            writer.add_scalar('Loss/test', te_loss, epoch)
    print("Done!")
    writer.close()

    