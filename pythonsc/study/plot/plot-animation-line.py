#学习python产生动画

#Sample command:
#python C:\fh\ws\ws1\fh-script\pythonsc\study\plot\plot-animation-line.py

from pylab import *
import numpy as np

#绘制图像，每隔0.5s画一个点，产生动画效果
for i in range(10):
    x = np.arange(-1*np.pi, np.pi, 0.1*(10-i+1))
    y_sin = np.sin(x)
    plt.cla() #清除再全部重画
    plt.plot(x, y_sin)
    plt.pause(0.5)
