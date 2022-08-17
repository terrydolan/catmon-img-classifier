# Catmon Image Classifier App: Datasets
This folder stores the *catmon_input* dataset, containing the image data.

This input is split into train, validation and test data - ready for modelling 
- to give the *catmon* dataset.
 
The instructions on how to download and prepare the dataset are given below.

## Create the *catmon_input* folder in the datasets folder
1. Set-up
 - cd to the root folder containing the *catmon-img-classifier* project files
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
 - extract contents of catmon-cats-dataset.zip to to ./datasets/
4. Verify
 - you should now have a folder ./datasets/catmon_input containing 3 folders:  
```
boo, simba, unknown
```
     - the *boo* folder contains 1000 images of Boo
     - the *simba* folder contains 1000 images of Simba
     - the *unknown* folder contains 793 images in which Boo or Simba cannot 
be identified

## Dataset Structure
The *catmon_input* dataset structure, sources and collection methodology 
is described on the [Kaggle catmon-cats-dataset](https://www.kaggle.com/datasets/terrydolan/catmon-cats-dataset).

## Splitting the *catmon_input* data ready for modelling

The *./datasets/catmon_input* is split to give the *./datasets/catmon* 
- a folder containing a training, validation and test images, ready for 
modelling - by running the 'Split Catmon Folders.ipynb'
notebook in the root folder.
