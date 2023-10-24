# Catmon Image Classifier

## Introduction
The *Catmon Image Classifier* (aka *Catmonic*) is a pytorch deep learning 
module that classifies a boosimba cat image with the cat's name and a 
probability.
The classifier returns a label and a probability; there are 3 possible labels: 
'boo', 'simba' or 'unknown'.

The model uses 'transfer learning' with a  pre-trained MobileNetV2 model 
applied to the catmon dataset; MobileNetV2 is pre-trained with millions of 
ImageNet images and so comes with cat recognition built in.
The catmon training and validation data is tuned over multiple epochs to find 
the most accurate cnn model.
Finally, the accuracy of the model is tested with an 'unseen' test dataset.

MobileNetV2 was selected because it has a small 'footprint', allowing the
application and model to be deployed on a standard raspberry pi.

## Catmonic Deployment
The catmonic module is now used to classify images as part of the *Catmon* app.
The app is deployed on a raspberry pi 3B+ and runs continuously, applying 
the catmonic deep learning CNN model to the catmon images.

## Catmon Output
An example automatic tweet from *Catmon* with a catmonic assisted classification:  
<img src="https://raw.githubusercontent.com/terrydolan/catmon-img-classifier/main/images/catmonic_classification_example_2023-10-24_134809.jpg" 
width="300">

## Catmonic Apps
The catmonic cli app, in the *catmonic-app* folder, uses catmonic to classify a given image.
There are some example images in the *images* folder.

The catmonic Twitter app was used before catmonic was embedded within *Catmon*.
The app processed the boosimba cat image that were attached to an auto-tweet 
from *Catmon*, classified the image and tweeted a reply with the cat's name.

## Datasets
The *catmon_input* folder containing the image data is available on Kaggle 
for download. 
See the *./datasets/DATASETS_README.md* file for more information on the 
structure of the image data and how to download.

## Key Project Files

The './catmonic/catmonic.py' is the main python implementation, a simple class 
that wrappers the catmon image classifier.

The './catmonic-app/catmonic_cli_app.py' is a simple python command line app that
classifies a given catmon image file using catmonic.

The './catmonic-app/catmonic_twitter_app2.py' is a python application that processes 
a tweet and classifies the catmon image file.

The logger configuration is in 'catmonic_logger.py' and the private twitter
data is in 'catmonic_twitter.ini' (not shared).

The 'Split Catmon Folders.ipynb' jupyter notebook splits the 
*./datasets/catmon_input* image data (sourced from the *Catmon Image Tagger* 
application) into the *./datasets/catmon* training, validation and test 
dataset. 

The 'Catmon Image Classifier Iteration3 mobilenet_v2.ipynb' notebook
generates the model from the *./datasets/catmon* dataset. 
The trained model is available in the *./catmonic/models* folder.

The 
'Classify Catmon Image From Twitter Stream Using MOBILENET_V2 CNN Model.ipynb'
was used to explore the solution ahead of producing the main application file.

There are additional notebooks that were used to prototype other aspects of
the solution.

## Usage
### Use The Catmonic Classifier
Example usage snippet:
```
import catmonic.catmonic as catmonic
# Instantiate the catmonic classifier object and classify the image
catmonic_clf = catmonic.Catmonic()
label, probability, model_name = (catmonic_clf.predict_catmon_image(pil_image))
```

### Run The Catmonic Apps
```
$python catmonic_cli_app.py
$python catmonic_twitter_app2.py
```
**Note that the Twitter app requires elevated twitter access to run the stream 
handler and so it is no longer active.**

However, it ran successfully for several months on a raspberry pi before Elon 
pulled the plug.

## Related Catmon Projects
1. *Catmon*: a cat flap monitor application that takes a picture when a cat
enters through the cat flap, tweets it on @boosimba and uploads the image
to google drive. 
The application ran continuously from 2015 to 2023 on a raspberry pi model B rev 2.
As part of the work to incorporate catmonic, *Catmon* was refactored and ported to a 
raspberry pi 3b+ running on a 64-bit aarch64 with Debian GNU/Linux 11 (bullseye).    
[Catmon repo](https://github.com/terrydolan/catmon)
2. *Catmon Image Tagger*: a web app that provides a UI to help tag a set of 
catmon images as either 'Boo' or 'Simba' or 'Unknown'.
The results are saved to google drive.  
[Catmon Image Tagger repo](https://github.com/terrydolan/catmon-img-tag)
3. *Catmon Last Seen*: an application that shows when Boo or Simba were 
last seen, using the output from *Catmon* and the *Catmon Image Classifier*.  
[Catmon Last Seen repo](https://github.com/terrydolan/catmon-lastseen)

Terry Dolan  
October 2023