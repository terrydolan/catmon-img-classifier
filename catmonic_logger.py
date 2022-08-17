"""Catmon image classifier logger configuration"""

import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# configure file handler
LOGFILE = 'catmonic_logger.log'
fhandler = RotatingFileHandler(
    filename=LOGFILE,
    maxBytes=1*1024*1024,
    backupCount=5
)
fformatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(fformatter)
fhandler.setLevel(logging.INFO)
logger.addHandler(fhandler)

# configure console handler
chandler = logging.StreamHandler()
cformatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
chandler.setFormatter(cformatter)
chandler.setLevel(logging.DEBUG)
logger.addHandler(chandler)
