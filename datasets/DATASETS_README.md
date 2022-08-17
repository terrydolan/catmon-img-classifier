# Catmon Image Classifier App: Datasets
This folder stores the *catmon_input* dataset, containing the image data.

This input is split into the *catmon* dataset, ready for modelling.

The instructions on how to download and split the datasets are given below.

## Create the *catmon_input* folder
1. Set-up
 - cd to the *root* folder containing the *catmon-img-classifier* project files
2. Download the dataset to the root folder
 - method 1: using the Kaggle CLI:  
```
kaggle datasets download terrydolan/catmon-cats-dataset
```
 - method 2: using Kaggle UI:
     - log on to Kaggle
     - select Datasets
     - search for 'Catmon Cats Dataset'
     - select the 'Download' option
3. Extract the images
 - extract contents of *catmon-cats-dataset.zip* to *./datasets/*
4. Verify
 - you should now have a folder *./datasets/catmon_input* containing 
 3 folders:  
```
boo, simba, unknown
```  
     - the *boo* folder contains 1000 images of Boo
     - the *simba* folder contains 1000 images of Simba
     - the *unknown* folder contains 793 images in which Boo or Simba
     Simba cannot be identified

## Dataset Structure
The *catmon_input* dataset structure, sources and collection methodology 
are described on the [Kaggle catmon-cats-dataset](https://www.kaggle.com/datasets/terrydolan/catmon-cats-dataset).

## Splitting the *catmon_input* data ready for modelling
The *./datasets/catmon_input* is split, ready for modelling, by running the 
*Split Catmon Folders.ipynb* notebook in the *root* folder.
This generates the *./datasets/catmon* folder containing 3 folders:  
```
train, val, test
```  
The split ratios are described in the notebook.
