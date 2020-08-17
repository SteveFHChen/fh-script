import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

import sys
import time
import json

mainScript = sys.argv[0]
mainPath = mainScript[:mainScript.rfind("/")+1]

sys.path.append(mainPath+'../utils')
import dbutils as db

param1=sys.argv[1]
#myprint_debug("param1: ", type(param1), param1);
params = json.loads(param1.replace("'", "\""));
