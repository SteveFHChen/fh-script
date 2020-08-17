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

#Sample command:
#success:
#python C:/fh/ws/ws1/fh-script/pythonsc/webcat/covidsklearn-lr.py "{'area': '美国', 'outputPath': 'C:/fh/testenv1/chart','diagramFileName':'plothello20200725-185829.png'}"

mainScript = sys.argv[0]
mainPath = mainScript[:mainScript.rfind("/")+1]

sys.path.append(mainPath+'../utils')
import dbutils as db

param1=sys.argv[1]
#myprint_debug("param1: ", type(param1), param1);
params = json.loads(param1.replace("'", "\""));

#import head

area=params['area'];
diagramFileName = params['diagramFileName']
picPath=params['outputPath'];

fileName = diagramFileName[:len(diagramFileName)-4]+'-2'+diagramFileName[-4:]

cs1 = db.conn.cursor()

input1 = 'covid_area_stat'
input2_2 = 'total_confirmed'
input2_1 = 'new_confirmed'
#input3 = '美国';
input3 = area;
input4 = "business_date"
input4_1 = "DATE_FORMAT(business_date, '%Y%m%d') business_date"


sql1 = "select %s from (%s) where area='%s' order by (%s) asc; " % (input4_1,input1,input3,input4)
cs1.execute(sql1)
alldata0 = cs1.fetchall()

bizdate = []
for s in alldata0:
    bizdate.append(s[0])
    
sql1 = "select (%s) from (%s) where area='%s' order by (%s) asc; " % (input2_1,input1,input3,input4)
cs1.execute(sql1)
total = []
alldata1 = cs1.fetchall()
for s in alldata1:
    total.append(s[0])

xidx = []
i=0
for s in bizdate:
    if (i>=1):
        if (bizdate[i-1]!=s):
            i = i+1
        xidx.append(i)
    if (i==0):
        xidx.append(i)
        i = i+1

xidx2d = []
for s in xidx:
    xidx2d.append([s])

xidx2d_future = []
xidx_last = xidx[len(xidx)-1]
for i in range(50):
    xidx2d_future.append([i+1+xidx_last])

poly_reg = PolynomialFeatures(degree=3)
xidx2d_poly = poly_reg.fit_transform(xidx2d)
xidx2d_future_poly = poly_reg.fit_transform(xidx2d_future)

#x_train, x_test, y_train, y_test = train_test_split(xidx2d, total, test_size=0.33, random_state=42)

model_LR = linear_model.LinearRegression()
model_LR.fit(xidx2d, total)
y_predict=model_LR.predict(xidx2d)
y_future_predict=model_LR.predict(xidx2d_future)

model_LR_poly = linear_model.LinearRegression()
model_LR_poly.fit(xidx2d_poly, total)
y_predict_poly=model_LR_poly.predict(xidx2d_poly)
y_future_predict_poly=model_LR_poly.predict(xidx2d_future_poly)

x_test = xidx2d
plt.subplot(1,1,1)

plt.plot(x_test, y_predict, label='LR', color='green')
#plt.scatter(x_test, y_predict, label='LR', color='green')

plt.plot(x_test, y_predict_poly, label='LR_poly', color='blue')
#plt.scatter(x_test, y_predict_poly, label='LR_poly', color='blue')

plt.plot(xidx2d_future, y_future_predict, label='Future', color='red')
plt.plot(xidx2d_future, y_future_predict_poly, label='Future', color='red')

plt.plot(xidx2d, total, label='Real', color='blue', linewidth=1.0, linestyle='dotted')
plt.scatter(xidx2d, total, label='Real', color='red')

plt.title('Compare Predict and Real Result')
plt.legend(loc='upper left')

plt.savefig(picPath+'/'+fileName,bbox_inches='tight')
#plt.show()

params['diagramFileName']=fileName
print(json.dumps(params))
