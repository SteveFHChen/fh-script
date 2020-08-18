
import sys

logLevel=3
# 1-debug, 2-log, 3-warning, 4-error

def getLogLevel():
    if logLevel == 4:
        return "error level"
    elif logLevel == 3:
        return "warning level"
    elif logLevel == 2:
        return "log level"
    elif logLevel == 1:
        return "debug level"
    else:
        return "unknow level: ", logLevel
        
def error(*s):
    #if logLevel <= 4:
        print("FH Error - ", s)
        
def warn(*s):
    if logLevel <= 3:
        print("FH Warn - ", s)
        
def log(*s):
    if logLevel <= 2:
        print("FH Log - ", s)
        
def debug(*s):
    if logLevel <= 1:
        print("FH Debug - ", s)

#How to use this module?
'''
import logger as lg

lg.error(lg.getLogLevel())
lg.logLevel=2
lg.error(lg.getLogLevel())

lg.debug("Hello debug")
lg.log("Hello log")
lg.warn("Hello warn")
lg.error("Hello error")
'''

#from logger import *
#This way cannot update variable logLevel in caller.
