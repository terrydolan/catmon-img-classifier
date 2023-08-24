#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Catmon image classifier (aka catmonic) Twitter app.

    Classify the boosimba cat image that is attached to a Twitter auto-tweet
    from catmon and reply with the cat's name.

    The boosimba tweet handler 'listens' on the @boosimba tweet stream for
    tweets. If the tweet is an auto-tweet from catmon it downloads the
    embedded cat image, runs the convolutional network model (cnn) model to
    classify the image.

    The classifier returns a label and a probability. There are 3 labels:
    'boo', 'simba' or 'unknown'. If the cat is identified then the handler
    tweets a reply with the cat's name and the probability of a successful
    classification.

    Author: Terry Dolan
    
    Note:
    - Twitter stream handler now requires paid access, so no longer running!

    References:
    -  For more information on the catmonic solution see the associated GitHub
    project.
    
    To Do:
        1. Refactor to use catmonic module (done, V2 June 2023)

"""
import os
import sys
from configparser import ConfigParser
from datetime import datetime
from io import BytesIO

import requests
import tweepy
from PIL import Image

sys.path.append("..")  # add parent folder to path
import catmonic.catmonic as catmonic  # import folder.file as myModule
from catmonic_logger import logger

# ----------------------------------------------------------------------------
__author__ = "Terry Dolan"
__maintainer__ = "Terry Dolan"
__copyright__ = "Terry Dolan"
__license__ = "MIT"
__email__ = "terry8dolan@gmail.com"
__status__ = "Beta"
__version__ = "0.2.1"
__updated__ = "August 2023"

# ----------------------------------------------------------------------------
# Set-up

# define handler config
DO_TWEET_REPLY = False
logger.debug(f"do tweet reply is: {DO_TWEET_REPLY}")

# define key twitter info
BOOSIMBA_TWITTER_CONFIG_FILE = 'catmonic_twitter.ini'
BOOSIMBA_TWITTER_USER_ID = 3022268453
logger.debug(f"boosimba titter config file is: {BOOSIMBA_TWITTER_CONFIG_FILE}")
logger.debug(f"boosimba twitter user id is: {BOOSIMBA_TWITTER_USER_ID}")

# ----------------------------------------------------------------------------
# define twitter helper function


def get_auth_info(TWITTER_CONFIG_FILE):
    """Return Twitter account name and auth info for twitter access."""

    config_path = os.path.abspath(TWITTER_CONFIG_FILE)

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Unexpected error, file not found: '{TWITTER_CONFIG_FILE}'"
        )

    # parse the config file and read key twitter info
    cfg = ConfigParser()
    cfg.read(config_path)

    account_name = cfg.get('twitter', 'account_name')
    consumer_key = cfg.get('twitter', 'consumer_key')
    consumer_secret = cfg.get('twitter', 'consumer_secret')
    access_token = cfg.get('twitter', 'access_token')
    access_token_secret = cfg.get('twitter', 'access_token_secret')

    auth_info = (
        consumer_key,
        consumer_secret,
        access_token,
        access_token_secret
    )

    return account_name, auth_info

# ----------------------------------------------------------------------------
# define twitter stream handler class


class BooSimbaTweetHandler(tweepy.Stream):
    """BooSimba tweet handler."""

    def __init__(
            self,
            do_tweet_reply,
            *auth_info,
            **kwargs
    ):
        # Instantiate classifier
        self.catmonic_clf = catmonic.Catmonic()
        logger.info(
            f"Initialise handler with catmonic classifier "
            f"(name={self.catmonic_clf.model_name}), "
            f"transform, device ({self.catmonic_clf.DEVICE}), "
            f"class_names ({self.catmonic_clf.CLASS_NAMES}) "
            f"do_tweet_reply ({do_tweet_reply}) and set-up twitter api "
            f"using given auth_info"
        )
        self.do_tweet_reply = do_tweet_reply
        auth = tweepy.OAuth1UserHandler(*auth_info)
        self.api = tweepy.API(auth)
        super().__init__(*auth_info, **kwargs)

    def _image_download(self, media_url):
        """Return downloaded pil image at given media_url."""
        response = requests.get(media_url)
        pil_img = Image.open(BytesIO(response.content))

        return pil_img

    def _reply_to_tweet(self, api, tweet_id, text):
        """Use api to post text as reply to tweet with given tweet_id."""
        try:
            api.update_status(
                    status=text,
                    in_reply_to_status_id=tweet_id,
                    auto_populate_reply_metadata=True
            )
        except TypeError as e:
            logger.error(
                f"Unexpected error replying on given api to tweet id "
                f"{tweet_id} with text '{text}' (error={e}, type={type(e)}"
            )
            raise

    def _handle_tweet(self, tweet):
        """Classify image in boosimba catmon auto-tweet."""
        logger.info(
            f"New tweet detected from {tweet.user.name} (user "
            f"id={tweet.user.id}) --------------------"
        )
        # logger.debug(tweet)
        BOOSIMBA_AUTOTWEET_TEXT = 'auto-tweet from catmon'
        BOOSIMBA_CAT_LABELS = ['boo', 'simba'] # excludes the 'unknown' label

        # get full text (in a different place if tweet is extended)
        if 'extended_tweet' in tweet._json:
            logger.debug('extended tweet')
            tweet_text = tweet.extended_tweet['full_text']
        else:
            tweet_text = tweet.text

        # create a single line of text, replacing \n
        tweet_text_sl = tweet_text.replace('\n', ' ')

        logger.info(f"Tweet text: {tweet_text_sl} (tweet id={tweet.id})")

        if tweet.user.id == BOOSIMBA_TWITTER_USER_ID and \
            tweet_text.startswith(BOOSIMBA_AUTOTWEET_TEXT):
            # all catmon auto-tweets should have an image, let's classify
            logger.info("Download image and classify...")
            try:
                # get media_url (in a different place if tweet is extended)
                if 'extended_tweet' in tweet._json:
                    media_url = tweet\
                        .extended_tweet['entities']['media'][0]['media_url']
                else:
                    media_url = tweet.entities['media'][0]['media_url']

                pil_image = self._image_download(media_url)

                # display half-size image (useful if running in jupyter)
                # w, h = pil_image.size
                # display(pil_image.resize((int(w/2), int(h/2))))

                # classify the image
                label, proba, model_name = self.catmonic_clf(pil_image)
                logger.info(
                    f"Classification: label={label}, probability={proba}, "
                    f"model_name={model_name}"
                )

                if label in BOOSIMBA_CAT_LABELS:
                    reply_text = (
                        f"Hello {label.capitalize()}\n\n\n"
                        f"[probability: {proba:.2%}, image automatically "
                        f"identified by the catmon image classifier cnn, "
                        f"using {self.model_name}]"
                    )

                    # create a single line of text, replacing \n
                    reply_text_sl = reply_text.replace('\n', ' ')

                    logger.info(f"Tweet reply text: {reply_text_sl}")
                    if self.do_tweet_reply:
                        self._reply_to_tweet(self.api, tweet.id, reply_text)
                    else:
                        logger.info("Tweeting of replies is switched off")

                else:  # label is 'unknown'
                    logger.info("Cat cannot be identified")
            except KeyError:
                logger.error("Unexpected error, could not access media")
        else:
            logger.info(
                "No action required, the tweet is not an auto-tweet from "
                "catmon (probably a reply)"
            )

    def on_status(self, status):
        """Handle status (tweet)."""
        self._handle_tweet(status)

    def on_error(self, status_code):
        """Handle error."""
        logger.error(f"Error detected: {status_code}")
        return True  # Don't kill the stream

    def on_timeout(self):
        """Handle timeout."""
        logger.warning("Timeout detected")
        return True  # Don't kill the stream


def main():
    """Main program."""
    logger.info(f"Started catmonic twitter app at {datetime.now()} =========")

    # get Twitter account name and auth info
    account_name, auth_info = get_auth_info(BOOSIMBA_TWITTER_CONFIG_FILE)
    assert account_name == 'boosimba', (
        f"Unexpected error, account_name '{account_name}' not recognised"
    )

    # instantiate the boosimba tweet stream handler to classify the
    # catmon auto-tweet images
    logger.info("Instantiate the BooSimbaTweetHandler")
    boosimba_tweet_handler = BooSimbaTweetHandler(
        DO_TWEET_REPLY,
        *auth_info
    )

    # run the handler, following the boosimba id
    logger.info(
        f"Run boosimba tweet handler (filter follow id "
        f"is: {BOOSIMBA_TWITTER_USER_ID})..."
    )
    boosimba_tweet_handler.filter(follow=[BOOSIMBA_TWITTER_USER_ID])


if __name__ == "__main__":
    main()
    