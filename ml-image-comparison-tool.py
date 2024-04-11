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
import generate_textures as gt
from scipy.stats import skew, kurtosis

from sklearnex import patch_sklearn
patch_sklearn()

#read images
from skimage.io import imread, imshow
from skimage.texture import graycomatrix, graycoprops
import matplotlib.pyplot as plt

#scikit-learn methods
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.inspection import PartialDependenceDisplay


#bayesian optimization
from skopt import BayesGridSearchCV

import shap
shap.initjs()


import mlflow
mlflow.autolog()
#run mlflow ui to start mlflow server and then http://localhost:5000


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


#%% #load images to dict for training

# Define the path to the directory containing the images
train_csv = r'image_path.csv'

def load_images(path):
    img = imread(path)
    return np.array(np.uint8(img))
 

with open(train_csv,"r") as f:
    data_dict = {'image': [], 'filename': [], 'label': [], 'entropy': [], 'otsu_segment': [], 'kmeans_segment': []}
    reader = csv.DictReader(f)
    for row in reader:
        image_path = row['path']
        label_name = row['label']

        data_dict['image'].append(load_images(image_path))   
        data_dict['filename'].append(image_path)   
        data_dict['label'].append(label_name) 
        data_dict['entropy'].append(gt.entropy_img(load_images(image_path),10))
        data_dict['otsu_segment'].append(gt.otsu_segment(load_images(image_path)))
        data_dict['kmeans_segment'].append(gt.kmeans_segment(load_images(image_path),5)) 


#prepare feature data for training
feature_df = pd.DataFrame(columns=['Image', 'Mean', 'Skew', 'Kurtosis', 'Variance', 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM', 'Target'])

# Loop over the images in the dictionary
for image_name, image in data_dict.items():
    #calculate GLCM textures
    glcm = greycomatrix(load_images(image_path), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # Calculate the statistical features and store them in the DataFrame
    feature_df = feature_df.append({
        'image': image_name,
        'mean': np.mean(load_images(image_path)),
        'skew': skew(load_images(image_path).flatten()),
        'kurtosis': kurtosis(load_images(image_path).flatten()),
        'variance': np.var(load_images(image_path)),
        'contrast': greycoprops(glcm, 'contrast')[0, 0], 
        'dissimilarity': greycoprops(glcm, 'dissimilarity')[0, 0], 
        'homogeneity': greycoprops(glcm, 'homogeneity')[0, 0], 
        'energy':greycoprops(glcm, 'energy')[0, 0], 
        'correlation': greycoprops(glcm, 'correlation')[0, 0], 
        'asm': greycoprops(glcm, 'ASM')[0, 0],
        'target': 'Target'
    }, ignore_index=True)


#%% Tune hyperparameters
params = {}
params['loss'] = ('log_loss', 'exponential')
params['n_estimators'] = (50, 100, 150, 200)
params['max_depth'] = (3, 5, 7, 9)
params['learning_rate'] = (0.01, 0.1, 0.5, 1.0)
params['min_samples_split'] = (2, 4, 6, 8)




#%% Classify image with classifiers

images = data_dict['image']
labels = data_dict['label']

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

images = [img.flatten() for img in images]
scaler = StandardScaler()
images = scaler.fit_transform(images)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# Define the classifier
gbc = GradientBoostingClassifier()

n_splits = 10
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

opt = BayesGridSearchCV(gbc, params, cv=cv)

shap_values = None

for fold, (train, test) in enumerate(cv.split(images, labels)):
    opt.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        opt,
        X_test,
        y_test,
        name=f"ROC fold {fold}",
        alpha=0.3,
        lw=1,
        ax=ax,
        plot_chance_level=(fold == n_splits - 1),
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    #SHAPley values
    if shap_values is None:
        explainer = shap.Explainer(opt.best_estimator_)
        shap_values = explainer(X_test)

    else:
        shap_values = np.vstack(
            [shap_values, explainer(X_test)]
        )


mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)


# Predict the labels of the test set
labels_pred = gpc.predict(images_test)


# %% Calculate metrics

shap.summary_plot(shap_values, X_test, plot_type='bar')


PartialDependenceDisplay.from_estimator(gbc, X_test, features, kind='individual')