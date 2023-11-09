########################
#
# Generate entropy and segmentation for images
#
# author: @mattlitz
#%% Load packages

import matplotlib.pyplot as plt
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

from skimage.io import imread, imshow
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import threshold_multiotsu

from sklearn.cluster import MiniBatchKMeans


#%% Generate image with entropy

#img = imread('images/C0015_0001.png')

#create entropy of image
def entropy_img(img, disk_size): 
    img_entropy = entropy(img, disk(disk_size))
    return img_entropy


#%% Generate segmented image using multiotsu


def otsu_segment(img):
    thresholds = threshold_multiotsu(img)
    img_otsu = np.digitize(img, bins=thresholds)
    return img_otsu



# %% Use Minibatch kmeans to segment out an image


def kmeans_segment(img, n_clusters):
    # Flatten the image
    pixels = img.reshape(-1, 1)

    # Perform MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_clusters)
    kmeans.fit(pixels)

    # Replace each pixel with its center
    segmented_img = kmeans.cluster_centers_[kmeans.predict(pixels)]
    segmented_img = segmented_img.reshape(img.shape)
    return segmented_img