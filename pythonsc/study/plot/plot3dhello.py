# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 14:04:47 2020

@author: steve
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

#1. 绘制3维的散点图==================>
x = np.random.randint(0,10,size=100)
y = np.random.randint(-20,20,size=100)
z = np.random.randint(0,30,size=100)
 
# 此处fig是二维
fig = plt.figure()
 
# 将二维转化为三维
axes3d = Axes3D(fig)
 
# axes3d.scatter3D(x,y,z)
# 效果相同
axes3d.scatter(x,y,z)

#2. 绘制三维的线性图==================>
x = np.linspace(0,20,1000)
 
y = np.sin(x)
z = np.cos(x)
 
fig = plt.figure(figsize=(8,6))
 
axes3d = Axes3D(fig)
 
axes3d.plot(x,y,z)
 
plt.xlabel('X',size = 30)
plt.ylabel('Y',size = 30)
axes3d.set_zlabel('Z',color = 'r',size=30)
 
axes3d.set_yticks([-1,0,1])
axes3d.set_yticklabels(['min',0,'max'])

#3. 绘制三维柱状图==================>
fig = plt.figure(figsize=(12,9))
 
axes3d = Axes3D(fig)
 
 
zs = [1,5,10,15,20]
 
for z in zs:
    x = np.arange(0,10)
    y = np.random.randint(0,30,size =10)
    axes3d.bar(x,y,zs = z,zdir = 'x',color=['r','green','yellow','purple'])

#4. 绘制三维曲面==================>
fig = plt.figure()
 
axes3d = Axes3D(fig)
 
#!！面
x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)
 
X,Y = np.meshgrid(x,y)
Z = np.sqrt(X**2+Y**2)

axes3d.plot_surface(X,Y,Z)

#5. 绘制等高线==================>
# 绘制面
from mpl_toolkits.mplot3d import axes3d
 
X,Y,Z = axes3d.get_test_data()
 
fig = plt.figure(figsize=(8,8))
axes3 = Axes3D(fig)
 
# 出现图形
axes3.plot_surface(X,Y,Z)
 
# 绘制等高线
axes3.contour(X,Y,Z,zdir = 'x',offset = -50)

plt.show()

