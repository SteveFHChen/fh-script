# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:55:09 2021

@author: steve
"""

#Approach 2: request data from API
#pip install PyExecJS
#3 approaches to get value from javascript
#Approach 1: using PyExecJS to execute the javascript, and get variable value;
#Approach 2: append reutrn command at the end of javascript, then using Selenium to start a browser and execute the javascript;
#Approach 3: cat the json string from the javascript, and use json.loads() to compile it to python dictionary;
#Approach 1 is the best.

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains

import os
import time

import requests
import json
import re
#import execjs

from pymysql import *

conn = connect(host='localhost', port=3306, database='test',
               user='root',
               password='root123', charset='utf8')
cs1 = conn.cursor()

driver = webdriver.Chrome(executable_path="C:/fh/testenv1/chromedriver.exe")
driver.set_page_load_timeout(40)

#Auto chat in wechat by program
try:
    driver.get("https://www.baidu.com") # get接受url可以是如何网址，此处以百度为例
except:
    print(f'{time.ctime()} Loading page timeout.')


# 浏览器头
headers = {'content-type': 'application/json',
           'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'}

#Using approach 1
def getCompiledJs(url):
    res = requests.get(url, headers=headers)
    content = res.text
    return execjs.compile(content)

#Using approach 3
def getJson(url, starti, endi):
    res = requests.get(url, headers=headers)
    content = res.text
    return json.loads(content[starti:-endi])
    
urlFundCompanyList = "http://fund.eastmoney.com/js/jjjz_gs.js"
fundCompanyList=getJson(urlFundCompanyList, len('var gs={op:'), len('}'))

for fc in fundCompanyList:
    sql = f"""
                INSERT INTO sec_fund_company(company_code, company_name) 
                VALUES('{fc[0]}', '{fc[1]}')
            """
    rowcount = cs1.execute(sql)
conn.commit()

urlFundList = "http://fund.eastmoney.com/js/fundcode_search.js"
fundList=getJson(urlFundList, len('var r = '), len(';'))

for f in fundList:
    sql = f"""
                INSERT INTO sec_fund(fund_code, fund_name, fund_type, short_pinyin) 
                VALUES('{f[0]}', '{f[2]}', '{f[3]}', '{f[1]}')
            """
    rowcount = cs1.execute(sql)
conn.commit()

fundCode = "010237" #"161725"
urlFund = "http://fund.eastmoney.com/pingzhongdata/%s.js"%fundCode
res = requests.get(urlFund, headers=headers)
content = res.text

#Python execute js script, and get variable value
fundInfo = execjs.compile(content)
netWorthTrend = fundInfo.eval('Data_netWorthTrend')

#funds=driver.execute_script(content)
#fundTrend=driver.execute_script(" return JSON.stringify(Data_netWorthTrend);")
#Cannot separate run, because they are in different session.

fundTrendList=json.loads(driver.execute_script(content+" return JSON.stringify(Data_netWorthTrend);"))

for ft in fundTrendList:
    sql = f"""
            INSERT INTO sec_fund_trend(net_date, fund_code, unit_net, day_delta_rate)
            VALUES({ft['x']}, '{fundCode}', {ft['y']}, {ft['equityReturn']})
        """
    rowcount = cs1.execute(sql)
conn.commit()

cs1.close()
conn.close()



#画图功能测试
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(11)-5
y=x*x

plt.subplot2grid((3,3), (0,0))
plt.plot(x, y)

plt.subplot2grid((3,3), (0,1))
plt.plot(x, y)

plt.subplot2grid((3,3), (1,1), colspan=2, rowspan=2)
plt.plot(x, y)

plt.show()











