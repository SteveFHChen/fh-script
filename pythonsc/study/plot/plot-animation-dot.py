#学习python产生动画

#Sample command:
#python C:\fh\ws\ws1\fh-script\pythonsc\study\plot\plot-animation-dot.py

from pylab import *
import numpy as np

x = np.arange(-1*np.pi, np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

#绘制图像，每隔0.1s画一个点，产生动画效果
#动画1：
'''
for i in range(x.size):
    plt.scatter(x[i], y_sin[i], label='LR', color='red')
    plt.scatter(x[i], y_cos[i], label='LR', color='green')
    plt.pause(0.05)
    '''
#动画2：
for i in range(x.size):
    plt.scatter(x[i], y_sin[i], label='LR', color='red')
    plt.scatter(x[x.size-i-1], y_cos[x.size-i-1], label='LR', color='green')
    plt.pause(0.05)
