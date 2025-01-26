from us_visa.logger import logging
from us_visa.exception import USvisaException
import sys

logging.info("Welcome to our visa project")

try:
    a=10
    a/0
except Exception as e:
    raise USvisaException(e, sys)