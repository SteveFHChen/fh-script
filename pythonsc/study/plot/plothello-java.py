import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import json

#Sample command:
#success:
#python C:/fh/ws1/fh-script/pythonsc/study/plothello-java.py {'outputPath':'C:/fh/pf/apache-tomcat-9.0.21/webapps/webcat/charts','diagramFileName':'-1'}
#python C:/fh/ws1/fh-script/pythonsc/study/plothello-java.py "{'outputPath': 'C:/fh/pf/apache-tomcat-9.0.21/webapps/webcat/charts','diagramFileName':'-1'}"
#
#Failed:
#python C:/fh/ws1/fh-script/pythonsc/study/plothello-java.py {'outputPath':'/C:/fh/pf/apache-tomcat-9.0.21/webapps/webcat/charts','diagramFileName':'-1'}

logLevel=20;
# 1-debug, 2-log, 3-warning, 4-error

def getLogLevel():
	if logLevel == 4:
		return "error level"
	elif logLevel == 3:
		return "warning level"
	elif logLevel == 2:
		return "log level"
	elif logLevel == 1:
		return "debug level"
	else:
		return "unknow level: ", logLevel
		
def myprint_error(*s):
	#if logLevel <= 4:
		print("Error - ", s)
		
def myprint_warn(*s):
	if logLevel <= 3:
		print("Warn - ", s)
		
def myprint_log(*s):
	if logLevel <= 2:
		print("Log - ", s)
		
def myprint_debug(*s):
	if logLevel <= 1:
		print("Debug - ", s)

#print("Hello world!");
#myprint_error(getLogLevel());

param1=sys.argv[1]
myprint_debug("param1: ", type(param1), param1);
params = json.loads(param1.replace("'", "\""));

picPath=params['outputPath'];
myprint_debug("picPath: "+picPath);

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
fileName='plothello'+t+'.png'
plt.savefig(picPath+'/'+fileName,bbox_inches='tight')

params['diagramFileName']=fileName;

print(json.dumps(params));
#plt.show()
