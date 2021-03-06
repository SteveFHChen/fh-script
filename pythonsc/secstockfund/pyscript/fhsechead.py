
print("Start to import fhsechead...")

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains

#import execjs
import js2py
#jsStock.eval(f"hq_str_{s2}") 中文乱码
#js2py.eval_js(line) 中文不乱码

import os
import os.path
import time
import datetime
import sys

def writeLog(msg):
    msgx = f'\n[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}][{os.getpid()}] {msg}'
    print(msgx)

writeLog("Start to import fhsechead...")

import math
import numpy as np
import matplotlib.pyplot as plt

import requests
import re
import json
import csv
import io

mainScript = sys.argv[0]
endSlashIndex = mainScript.rfind("/", 0, mainScript.rfind("/")-1)
mainPath = mainScript[:endSlashIndex+1]
writeLog(f"mainScript: [{mainScript}]")
writeLog(f"mainPath: [{mainPath}]")

#sys.path.append(mainPath+'../utils')
sys.path.append('C:/fh/ws/ws1/fh-script/pythonsc/utils')
import logger as lg
from LogWriter import *
from properties import *
#import dbutils as db
from dbutils import *
from rank import *

#Web UI capture, skip loading image to improve performance
chrome_options = webdriver.ChromeOptions()
prefs = {"profile.managed_default_content_settings.images":2}
chrome_options.add_experimental_option("prefs",prefs)

#HTTP API access
headers = {'content-type': 'application/json',
           'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'}

chromeDriverExe="C:/fh/testenv1/chromedriver.exe"

writeLog("Import fhsechead completed.")

g_interval_days = 50