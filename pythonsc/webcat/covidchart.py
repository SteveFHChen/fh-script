#coding: UTF-8

#Sample command:
#python C:/fh/ws/ws1/fh-script/pythonsc/webcat/covidchart.py "{'area': '美国', 'outputPath': 'C:/fh/pf/apache-tomcat-9.0.21/webapps/webcat/charts','diagramFileName':'-1'}"

from head import *

#myprint_error(getLogLevel());
#lg.logLevel=1

area=params['area'];
picPath=params['outputPath'];
lg.debug("picPath: "+picPath);

#Testing parameters
#picPath="D:\\pythontest";
#print("1 picPath: ", picPath);
#print("2 picPath: "+picPath);

cs1 = db.conn.cursor()
 #四个参数，表名，列名1，列名2，预测年份
input1 = 'covid_area_stat'
input2_1 = 'total_confirmed'
input2_2 = 'new_confirmed'
input3 = area
input4 = 'business_date'

#读取第一列year
sql1 = "select (%s) from (%s) where area='%s' order by (%s) asc; " % (input4,input1,input3,input4)
cs1.execute(sql1)
datalist0 = []
alldata0 = cs1.fetchall()
for s in alldata0:
    datalist0.append(s[0])
lg.debug(datalist0)

sql1 = "select (%s) from (%s) where area='%s' order by (%s) asc; " % (input2_1,input1,input3,input4)
cs1.execute(sql1)
datalist1 = []
alldata1 = cs1.fetchall()
for s in alldata1:
    datalist1.append(s[0])
lg.debug(datalist1)

sql1 = "select (%s) from (%s) where area='%s' order by (%s) asc; " % (input2_2,input1,input3,input4)
cs1.execute(sql1)
datalist2 = []
alldata2 = cs1.fetchall()
for s in alldata2:
    datalist2.append(s[0])
lg.debug(datalist2)


#新版的sklearn中，所有的数据都应该是二维矩阵，哪怕它只是单独一行或一列（比如前面做预测时，仅仅只用了一个样本数据），所以需要使用.reshape(1,-1)进行转换
 #datalist11=np.array(datalist1).reshape(len(datalist1),-1)
datalist11=[]
for i in datalist1:
    list1=[i]
    datalist11.append(list1)
#datalist22=np.array(datalist2).reshape(1,-1)
lg.debug(datalist11)

x=np.arange(len(datalist11))

#display total case
plt.subplot(1,2,1);
plt.plot(datalist0, datalist1, label='Sin(x)', color='blue', linewidth=2.0, linestyle='dotted');
plt.scatter(datalist0, datalist1, label='Sin(x)', color='red');

plt.title(f"{area}-累计确诊人数", fontproperties=font);

#display new case
plt.subplot(1,2,2);
plt.plot(datalist0, datalist2, label='Sin(x)', color='blue', linewidth=2.0, linestyle='dotted');
plt.scatter(datalist0, datalist2, label='Sin(x)', color='red');
plt.title(f"{area}-新增确诊人数", fontproperties=font);

t=time.strftime("%Y%m%d-%H%M%S", time.localtime());
fileName='plothello'+t+'.png'

plt.savefig(picPath+'/'+fileName,bbox_inches='tight')
#plt.show()

params['diagramFileName']=fileName;
print(json.dumps(params));
