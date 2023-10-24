#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Catmon image classifier module.

    Provide a simple class that wrappers the catmon image classifier, also
    known as catmonic.


    Catmonic instantiates the pre-build MobileNetV2 model and provides a
    predict_catmon_image() method to classify the given catmon image.

    The classifier returns the label, the probability and the model name.

    There are 3 labels:
    'boo', 'simba' or 'unknown'. If the cat is identified then the handler
    tweets a reply with the cat's name and the probability of a successful
    classification.

    Author: Terry Dolan

    Note:
    1. catmonic was originally embedded within a standalone tweet handler
    application. It has been moved to a module to allow it to be included in
    other apps.

    References:
    - For more information on the catmon* solution see the associated GitHub
    projects.

    To Do:

    Changes:
    1. 24th October 2023: Use 'importlib.resources' to improve handling of model file in package

"""

# ----------------------------------------------------------------------------
# define code meta data
__author__ = "Terry Dolan"
__maintainer__ = "Terry Dolan"
__copyright__ = "Terry Dolan"
__license__ = "MIT"
__email__ = "terry8dolan@gmail.com"
__status__ = "Beta"
__version__ = "0.1.1"
__updated__ = "October 2023"

# ----------------------------------------------------------------------------
# define imports
from importlib.resources import files

from torch import nn
from torchvision import transforms, models
import torch


# ----------------------------------------------------------------------------
# define Catmonic class


class Catmonic:
    # define class wide variables

    # define model class names
    CLASS_NAMES = ['boo', 'simba', 'unknown']

    # define image transform
    # - as used during model transfer learning, training and test
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
        ])

    # define device
    DEVICE = "cpu"

    # define location of model state dict containing key paramaters
    MODEL_SD = files('catmonic.models').joinpath('catmon-img-classifier_mobilenet_v2_state_dict_0.3')

    def __init__(self):
        # prepare the model
        self.model = self._prepare_mobilenet_v2_model()
        self.model_name = self.model.__class__.__name__

    def predict_catmon_image(
            self,
            pil_image):
        """Transform pil image, apply classification model and return the
        predicted label, probability and model name."""

        # apply transform to create the required tensor
        tensor = self.TRANSFORM(pil_image)

        # reshape tensor to add dummy batch
        reshaped_tensor = tensor.unsqueeze(0)

        # classify the image
        with torch.no_grad():
            inputs = reshaped_tensor.to(self.DEVICE)
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)

            # calculate probabilities using softmax
            sm = torch.nn.Softmax(dim=1)
            probabilities = sm(outputs)

            label = self.CLASS_NAMES[preds]
            probability = float(probabilities.flatten()[preds])

        return label, probability, self.model_name

    def _prepare_mobilenet_v2_model(self):
        """Prepare mobilenet_v2 model for evaluation."""
        # instantiate the model
        model = models.mobilenet_v2()

        # extract number of input and output features from the model
        num_features = model.classifier[-1].in_features

        # update model classifier to take account of number of classes
        num_classes = len(self.CLASS_NAMES)
        model.classifier[-1] = nn.Linear(num_features, num_classes)

        # load state dict for the trained model
        model.load_state_dict(torch.load(self.MODEL_SD))

        # place model in evaluation mode
        model.eval()

        return model
