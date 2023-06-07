# Catmon Image Classifier App

## Introduction
The *Catmon Image Classifier* (aka *Catmonic*) is a pytorch deep learning 
module that classifies a boosimba cat image with the cat's name and a probability.

The model uses 'transfer learning' with a  pre-trained MobileNetV2 model 
applied to the catmon dataset; MobileNetV2 is pre-trained with millions of 
ImageNet images and so comes with cat recognition built in.
The catmon training and validation data is tuned over multiple epochs to find 
the most accurate cnn model.
Finally the accuracy of the model is tested with an 'unseen' test dataset.

MobileNetV2 was selected because it has a small 'footprint', allowing the
application and model to be deployed on a standard raspberry pi.

## Apps
The catmonic twitter app uses the catmonic image classifier to process the boosimba 
cat image that is attached to an auto-tweet from *Catmon* and replies with the cat's 
name.

The app's tweet handler 'listens' to the @boosimba tweet stream for tweets. 
If the tweet is an auto-tweet from *Catmon* it downloads the embedded cat 
image and runs the trained convolutional network model (cnn) model to 
classify the image.
    
The classifier returns a label and a probability; there are 3 possible 
labels: 'boo', 'simba' or 'unknown'. 
If the cat is identified then the handler tweets a reply with the cat's name 
and the probability of a successful classification.

## Datasets
The *catmon_input* folder containing the image data is available on Kaggle 
for download. 
See the *./datasets/DATASETS_README.md* file for more information on the 
structure of the image data and how to download.

## Key Project Files
The './apps/catmonic_twitter_app2.py' is the main python application.
The logger configuration is in 'catmonic\_logger.py' and the private twitter
data is in 'catmonic\_twitter.ini' (not shared).

**Note that this app requires elevated twitter access and so it is 
no longer active.**

The './apps/catmonic_cli_app.py' is a simple python command line app that
classifies a given catmon image file.

The 'Split Catmon Folders.ipynb' jupyter notebook splits the 
*./datasets/catmon_input* image data (sourced from the *Catmon Image Tagger* 
application) into the *./datasets/catmon*) training, validation and test 
dataset. 

The 'Catmon Image Classifier Iteration3 mobilenet\_v2.ipynb' notebook
generates the model from the *./datasets/catmon* dataset. 
The trained model is available in the *models* folder.

The 
'Classify Catmon Image From Twitter Stream Using MOBILENET\_V2 CNN Model.ipynb'
was used to explore the solution ahead of producing the main application file.

There are additional notebooks that were used to prototype other aspects of
the solution.

## Catmonic Deployment
The app is deployed on a raspberry pi 3B+, configured with: 
python, requests, PIL, 
pytorch, torchvision, and tweepy.
It runs continuously, applying the deep learning CNN model to the catmon 
images.

### Run the apps
```
$python catmonic_twitter_app2.py
$python catmonic_cli_app.py
```

## Catmonic Output
The output from the running app can be seen as a reply to a catmon auto-tweet 
on the [@boosimba twitter account](https://twitter.com/boosimba).


An example classification:  
<img src="https://raw.githubusercontent.com/terrydolan/catmon-img-classifier/main/images/catmonic_classification_example_2022-08-08_083902.jpg" 
width="300">

## Related Catmon Projects
1. *Catmon*: a cat flap monitor application that takes a picture when a cat
enters through the cat flap, tweets it on @boosimba and uploads the image
to google drive. 
The application has been running since 2015 on a raspberry pi model B rev 2.  
[Catmon repo](https://github.com/terrydolan/catmon)
1. *Catmon Image Tagger*: a web app that provides a UI to help tag a set of 
catmon images as either 'Boo' or 'Simba' or 'Unknown'.
The results are saved to google drive.  
[Catmon Image Tagger repo](https://github.com/terrydolan/catmon-img-tag)
1. *Catmon Last Seen*: an application that shows when Boo or Simba were 
last seen, using the output from *Catmon* and the *Catmon Image Classifier*.  
[Catmon Last Seen repo](https://github.com/terrydolan/catmon-lastseen)

Terry Dolan  
June 2023