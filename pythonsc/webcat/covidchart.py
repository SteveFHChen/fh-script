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

sql1 = f"""
	select 
		DATE_FORMAT(business_date, '%Y%m%d') business_date, 
		total_confirmed, new_confirmed 
	from covid_area_stat 
	where area='{area}' 
	order by business_date asc
	"""
lg.debug(sql1)

cs1.execute(sql1)

bizdate = []
totalConfirm = []
newConfirm = []
records = cs1.fetchall()
for s in records:
    bizdate.append(s[0])
    totalConfirm.append(s[1])
    newConfirm.append(s[2])

#新版的sklearn中，所有的数据都应该是二维矩阵，哪怕它只是单独一行或一列（比如前面做预测时，仅仅只用了一个样本数据），所以需要使用.reshape(1,-1)进行转换
 #datalist11=np.array(datalist1).reshape(len(datalist1),-1)
totalConfirm1=[]
for i in totalConfirm:
    list1=[i]
    totalConfirm1.append(list1)
lg.debug(totalConfirm1)

bizdate_dist = distinct(bizdate)
x=[]
xlabel=[]
for i in range(len(bizdate_dist)):
    if(i%10==0):
        x.append(i)
        xlabel.append(bizdate_dist[i][4:])
#print('len(bizdate):', len(bizdate))
#print('x:', x)

#display total case
plt.subplot(1,2,1);
plt.plot(bizdate, totalConfirm, color='blue', linewidth=2.0, linestyle='dotted');
plt.scatter(bizdate, totalConfirm, color='red');

plt.title(f"{area}-累计确诊人数", fontproperties=font);
plt.xticks(x,xlabel)

#display new case
plt.subplot(1,2,2);
plt.plot(bizdate, newConfirm, color='blue', linewidth=2.0, linestyle='dotted');
plt.scatter(bizdate, newConfirm, color='red');

plt.title(f"{area}-新增确诊人数", fontproperties=font);
plt.xticks(x,xlabel)

t=time.strftime("%Y%m%d-%H%M%S", time.localtime());
fileName='plothello'+t+'.png'

plt.savefig(picPath+'/'+fileName,bbox_inches='tight')
#plt.show()

params['diagramFileName']=fileName;
print(json.dumps(params));
