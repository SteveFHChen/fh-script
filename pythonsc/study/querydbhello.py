from pymysql import *
#import pandas as pd
#import numpy as np
#from sklearn import linear_model
#from sqlalchemy import create_engine
#import matplotlib.pyplot as plt


conn = connect(host='localhost', port=3306, database='test',
               user='root',
               password='root123', charset='utf8')
cs1 = conn.cursor()
 #四个参数，表名，列名1，列名2，预测年份
input1 = 'covid_area_stat'
input2 = 'total_confirmed'
input3 = '美国'
input4 = 'business_date'

#读取第一列year
sql1 = "select (%s) from (%s) where area='%s' order by (%s) desc; " % (input2,input1,input3,input4)
cs1.execute(sql1)
datalist1 = []
alldata1 = cs1.fetchall()
for s in alldata1:
    datalist1.append(s[0])
print(datalist1)


#新版的sklearn中，所有的数据都应该是二维矩阵，哪怕它只是单独一行或一列（比如前面做预测时，仅仅只用了一个样本数据），所以需要使用.reshape(1,-1)进行转换
 #datalist11=np.array(datalist1).reshape(len(datalist1),-1)
datalist11=[]
for i in datalist1:
    list1=[i]
    datalist11.append(list1)
#datalist22=np.array(datalist2).reshape(1,-1)
print(datalist11)

