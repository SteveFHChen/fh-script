import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import sys
import time
import json

#Sample command:
#success:
#python C:/fh/ws/ws1/fh-script/pythonsc/webcat/covidsklearn.py "{'area': '美国', 'outputPath': 'C:/fh/testenv1/chart','diagramFileName':'plothello20200725-185829.png'}"

mainScript = sys.argv[0]
mainPath = mainScript[:mainScript.rfind("/")+1]

sys.path.append(mainPath+'../utils')
import dbutils as db

param1=sys.argv[1]
#myprint_debug("param1: ", type(param1), param1);
params = json.loads(param1.replace("'", "\""));

area=params['area'];
diagramFileName = params['diagramFileName']
picPath=params['outputPath'];

fileName = diagramFileName[:len(diagramFileName)-4]+'-2'+diagramFileName[-4:]

cs1 = db.conn.cursor()
 #四个参数，表名，列名1，列名2，预测年份
input1 = 'covid_area_stat'
input2_1 = 'total_confirmed'
input2_2 = 'new_confirmed'
#input3 = '南非'
input3 = area;
input4 = "business_date"
input4_1 = "DATE_FORMAT(business_date, '%Y%m%d') business_date"

#读取第一列year
sql1 = "select %s from (%s) where area='%s' order by (%s) asc; " % (input4_1,input1,input3,input4)
cs1.execute(sql1)
datalist0 = []
datalist02 = []
alldata0 = cs1.fetchall()
for s in alldata0:
    datalist0.append(s[0])
    datalist02.append([s[0]])


sql1 = "select (%s) from (%s) where area='%s' order by (%s) asc; " % (input2_1,input1,input3,input4)
cs1.execute(sql1)
datalist1 = []
alldata1 = cs1.fetchall()
for s in alldata1:
    datalist1.append(s[0])


sql1 = "select (%s) from (%s) where area='%s' order by (%s) asc; " % (input2_2,input1,input3,input4)
cs1.execute(sql1)
datalist2 = []
alldata2 = cs1.fetchall()
for s in alldata2:
    datalist2.append(s[0])

x=datalist02
y=datalist2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

x_train2=[]
for s in x_train:
    x_train2.append(s[0])

#Define model
knn=KNeighborsClassifier()

#Train model with the trainning data
knn.fit(x_train, y_train)

#Predict data with the trainned model
#y_predict=knn.predict(x_test)
y_predict=knn.predict(x)

#xindex = np.arange(len(y_predict));
xindex = datalist0

'''
plt.subplot(2,2,1);
plt.plot(xindex, y_predict, label='Predict', color='green');
plt.title('Predict Result');
plt.legend(loc='upper left')

plt.subplot(2,2,2);
plt.plot(xindex, y_test, label='Line', color='blue', linewidth=2.0, linestyle='dotted');
plt.scatter(xindex, y_test, label='Real', color='red');
plt.title('Real Result');
plt.legend(loc='upper left')
'''

#plt.subplot(2,2,3);
plt.subplot(1,1,1);

plt.plot(xindex, y_predict, label='Predict', color='green');
plt.scatter(xindex, y_predict, label='Predict', color='green');

#plt.plot(xindex, y_test, label='Line', color='blue', linewidth=2.0, linestyle='dotted');
#plt.scatter(xindex, y_test, label='Real', color='red');

plt.plot(xindex, y, label='Line', color='blue', linewidth=2.0, linestyle='dotted');
plt.scatter(xindex, y, label='Real', color='red');

plt.scatter(x_train2, y_train, s=10, label='train', color='yellow');


plt.title('Compare Predict and Real Result');
plt.legend(loc='upper left')

plt.savefig(picPath+'/'+fileName,bbox_inches='tight')
#plt.show()

params['diagramFileName']=fileName;
print(json.dumps(params));
