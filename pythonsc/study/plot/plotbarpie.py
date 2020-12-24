# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 11:08:12 2020

@author: steve
Refer: 
    python 使用 matplotlib.pyplot来画柱状图和饼图
    https://www.cnblogs.com/zhhfan/p/9971757.html
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
#font = FontProperties(fname=mainPath+"simsun.ttc", size=10) #字体设置方式1，可显示中文字体
font = {'family':'SimHei', 'weight':'normal', 'size':10} #字体设置方式2，可显示中文字体

expl = [0, 0, 0.3, 0, 0, 0.2, 0.1]
num_list = [33, 24, 15, 66, 77, 22, 11]
num_list2 = np.array(num_list)+5
lab_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
x = list(range(len(num_list)))

plt.subplot(2,2,1);
#plt.bar(num_list, num_list, tick_label=lab_list)
plt.bar(x, num_list, color='rgb', tick_label=lab_list, width=0.5)
plt.bar(x, num_list, color='y', tick_label=lab_list, width=0.4, bottom=np.array(num_list)-10)
plt.plot(x, num_list)

plt.subplot(2,2,2);
plt.bar(x, num_list, color='g', tick_label=lab_list, label='2019年', width=0.4)
plt.bar(np.array(x)+0.45, num_list2, color='b', tick_label=lab_list, label='2020年', width=0.4)
plt.legend(prop=font)

plt.subplot(2,2,3);
plt.pie(x, labels=lab_list, autopct='%3.1f %%')

plt.subplot(2,2,4);
plt.pie(x, explode=expl, labels=lab_list, autopct='%3.1f %%', shadow=True) 

#plt.text(42, 40, '44')
#plt.text(55, 55, '55')


plt.show()
