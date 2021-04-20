import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
import sklearn.metrics as ms
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties


from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

import sys
import time
import json

mainScript = sys.argv[0]
mainPath = mainScript[:mainScript.rfind("/")+1]

font = FontProperties(fname=mainPath+"simsun.ttc", size=10)
sys.path.append(mainPath+'../utils')
import logger as lg
import dbutils as db
from rank import *

param1=sys.argv[1]
#myprint_debug("param1: ", type(param1), param1);
params = json.loads(param1.replace("'", "\""));

#====================Description==================================
#How to use?
#Way 1 - import all with namespace:
#import head
#import head as h
#Import the libs, variables and function or class, but in a namespace.

#Way 2 - import parts or all without namespace:
#from head import *
#Equals copy the block source code to there, so including the import libs.
#Imported into a non-name space, so can use directly, but cannot update variable.
#Also can import one function from the module.

#Way 3 - just run the python script:
#code=os.system("python head.py ...")
