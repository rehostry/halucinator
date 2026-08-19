# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC 
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, 
# the U.S. Government retains certain rights in this software.

'''
Uses config file format to control logging, first looks in local 
directory for config file and uses it if set, else uses the
default on from halucinator

For file format see: https://docs.python.org/3/library/logging.config.html#logging-config-fileformat
'''
import logging
import logging.config
from os import path


LOG_CONFIG_NAME = 'logging.cfg'
DEFAULT_LOG_CONFIG = path.join(path.dirname(__file__),'logging.cfg')
HAL_LOGGER = "HAL_LOG"

def getHalLogger():
    return logging.getLogger(HAL_LOGGER)

def setLogConfig():
    hal_log = getHalLogger()
    if path.isfile(LOG_CONFIG_NAME):
        hal_log.info("USING LOGGING CONFIG From: %s" % LOG_CONFIG_NAME)
        # disable_existing_loggers MUST match the default-config branch below.
        # Every halucinator module does `log = logging.getLogger(__name__)` at
        # IMPORT time, i.e. before setLogConfig() runs, so those loggers are
        # all "existing" by the time this executes. With True, dropping a
        # logging.cfg into the working directory silenced every `halucinator.*`
        # logger in the process -- the run still worked, you just went blind,
        # and a broken peripheral model (see _x86_in_hook, which logs a warning
        # and then returns 0) became completely silent. The packaged-default
        # branch has always used False; this asymmetry was the bug.
        logging.config.fileConfig(fname=LOG_CONFIG_NAME, disable_existing_loggers=False)
    else:  # Default logging
        hal_log.info("USING DEFAULT LOGGING CONFIG")
        hal_log.info("This behavior can be overwritten by defining %s"% LOG_CONFIG_NAME)
        logging.config.fileConfig(fname=DEFAULT_LOG_CONFIG, disable_existing_loggers=False)
