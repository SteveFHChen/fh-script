import numpy as np
import matplotlib.pyplot as plt
import sys
import time

print("Hello world!");

for i in range(1, len(sys.argv)):
	if (i==1):
		picPath=sys.argv[i]
	print(sys.argv[i]);

print("picPath: "+picPath);

#Testing parameters
#picPath="D:\\pythontest";
#print("1 picPath: ", picPath);
#print("2 picPath: "+picPath);

x = np.arange(-1*np.pi, np.pi, 0.1);

y_sin = np.sin(x);
y_cos = np.cos(x);
y_tan = np.tan(x);
y_arctan = np.arctan(x);

plt.subplot(2,2,1);
#Label用于图例
#linestyle supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
plt.plot(x, y_sin, label='Sin(x)', color='red', linewidth=2.0, linestyle='dotted');
plt.plot(x, y_cos, label='Cos(x)', color='green', linewidth=3.0, linestyle='dashdot');
plt.title('Sin');
#增加图例
plt.legend(loc='upper left');

plt.subplot(2,2,2);
plt.plot(x, y_cos)
plt.title('Cos');

#设置刻度值标签
plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],[r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'])
plt.yticks([-1,0,1],[r'$-1$',r'$0$',r'$1$'])

#移动spines, 移动边界线，构建坐标系，原点为0
ax = plt.gca()#获取当前轴线实例
ax.xaxis.set_ticks_position('bottom')#x轴线，使用spine中的bottom线
ax.yaxis.set_ticks_position('left')#y轴线，使用spine中的left线
ax.spines['bottom'].set_position(('data',0))#将bottom线的位置设置为数据为0的位置
ax.spines['left'].set_position(('data',0))#将left线的位置设置为数据为0的位置
ax.spines['top'].set_color('none')#将top线的颜色设置为无
ax.spines['right'].set_color('none')#将right线的颜色设置为无



plt.subplot(2,2,3);
plt.plot(x, y_tan);
plt.title('Tan');

plt.subplot(2,2,4);
plt.plot(x, y_arctan);
plt.title('Arctan');

t=time.strftime("%Y%m%d-%H%M%S", time.localtime());
plt.savefig(picPath+'\\plothello'+t+'.png',bbox_inches='tight')

#plt.show()
