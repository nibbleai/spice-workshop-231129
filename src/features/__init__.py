import logging

from spice import SpiceLoggingHandler

logger = logging.getLogger()
logger.addHandler(SpiceLoggingHandler())
logger.setLevel(logging.DEBUG)
