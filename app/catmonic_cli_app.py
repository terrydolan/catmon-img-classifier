#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Catmon image classifier cli app.

    The app provides a simple command-line interface that classifies the
    given (catmon) image using catmonic, the catmon image classifier module.


    Catmonic runs a pre-built convolutional network model (cnn) model to
    classify the image. The classifier returns the label, the probability and
    the model name. There are 3 labels: 'boo', 'simba' or 'unknown'.

    The app is typically used with images from catmon. These can be seen on
    the @boosimba Twitter account.

    Author: Terry Dolan

    References:
    - For more information on the catmon* solution see the associated GitHub
    projects.

    To Do:
        1. Support url of catmon image as input.

"""

# ----------------------------------------------------------------------------
# Define code meta data
__author__ = "Terry Dolan"
__maintainer__ = "Terry Dolan"
__copyright__ = "Terry Dolan"
__license__ = "MIT"
__email__ = "terry8dolan@gmail.com"
__status__ = "Beta"
__version__ = "0.1.1"
__updated__ = "August 2023"

# ----------------------------------------------------------------------------
# Define imports
import sys

import argparse
from PIL import Image

sys.path.append("..")  # add parent folder to path
import catmonic.catmonic as catmonic  # import folder.file as myModule


# ----------------------------------------------------------------------------
# Define helper functions


def open_image(image_path):
    """Return PIL image for given image file path."""
    try:
        image = Image.open(image_path)
        return image
    except IOError as e:
        print(f"Error opening image: {e}")
        return None

# ----------------------------------------------------------------------------
# Define main function


def main():
    """Classify given image path using catmonic."""
    # Create the command-line argument parser
    parser = argparse.ArgumentParser(
        prog='catmonic_cli_app.py',
        description='Catmonic Command Line Interface App')

    # Add the image path argument
    parser.add_argument(
        'image_path', type=str, help='Path to the input image')

    # Parse the command-line arguments, collect the image path
    args = parser.parse_args()

    # Open the PIL image using the given image path
    pil_image = open_image(args.image_path)

    if pil_image:
        # Instantiate the catmonic classifier object and classify the image
        catmonic_clf = catmonic.Catmonic()
        label, probability, model_name = (
            catmonic_clf.predict_catmon_image(pil_image))

        # show the classification: the label and probability
        print(f"{label.capitalize()}, {probability:.2%} ({model_name})")


if __name__ == '__main__':
    main()
