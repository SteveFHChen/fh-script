
#Sample command:
#success:
#python C:/fh/ws/ws1/fh-script/pythonsc/webcat/covidsklearn-lr.py "{'area': '美国', 'outputPath': 'C:/fh/testenv1/chart','diagramFileName':'plothello20200725-185829.png'}"

from head import *
#lg.logLevel=1

area=params['area'];
diagramFileName = params['diagramFileName']
picPath=params['outputPath'];

fileName = diagramFileName[:len(diagramFileName)-4]+'-2'+diagramFileName[-4:]

cs1 = db.conn.cursor()

sql1 = f"""
	select 
		DATE_FORMAT(business_date, '%Y%m%d') business_date, new_confirmed 
	from covid_area_stat 
	where area='{area}' 
	order by business_date asc
	"""
lg.debug(sql1)

cs1.execute(sql1)
alldata0 = cs1.fetchall()

bizdate = []
total = []
for s in alldata0:
    bizdate.append(s[0])
    total.append(s[1])
    
xidx = dense_rank(bizdate)

bizdate_dist = distinct(bizdate)
x=[]
xlabel=[]
for i in range(len(bizdate_dist)):
    if(i%10==0):
        x.append(i)
        xlabel.append(bizdate_dist[i][4:])
#print('len(bizdate):', len(bizdate))
#print('x:', x)

xidx2d = []
for s in xidx:
    xidx2d.append([s])

xidx2d_future = []
bizdate_future = []
xidx_last = xidx[len(xidx)-1]
for i in range(20):
    xidx2d_future.append([i+1+xidx_last])

poly_reg = PolynomialFeatures(degree=3)
xidx2d_poly = poly_reg.fit_transform(xidx2d)
xidx2d_future_poly = poly_reg.fit_transform(xidx2d_future)

#x_train, x_test, y_train, y_test = train_test_split(xidx2d, total, test_size=0.33, random_state=42)

model_LR = linear_model.LinearRegression()
model_LR.fit(xidx2d, total)
y_predict=model_LR.predict(xidx2d)
y_future_predict=model_LR.predict(xidx2d_future)

model_LR_poly = linear_model.LinearRegression()
model_LR_poly.fit(xidx2d_poly, total)
y_predict_poly=model_LR_poly.predict(xidx2d_poly)
y_future_predict_poly=model_LR_poly.predict(xidx2d_future_poly)

p=0.3
y_predict_avg = []
for i in range(len(y_predict)):
    y_predict_avg.append(p*y_predict[i]+(1-p)*y_predict_poly[i])

y_future_predict_avg = []
for i in range(len(y_future_predict)):
    y_future_predict_avg.append(p*y_future_predict[i]+(1-p)*y_future_predict_poly[i])

x_test = xidx2d
plt.subplot(1,1,1)

plt.plot(x_test, y_predict, label='LR', color='green')
#plt.scatter(x_test, y_predict, label='LR', color='green')

plt.plot(x_test, y_predict_poly, label='LR_poly', color='blue')
#plt.scatter(x_test, y_predict_poly, label='LR_poly', color='blue')

plt.plot(x_test, y_predict_avg, label='LR_avg', color='deeppink')

plt.plot(xidx2d_future, y_future_predict, label='Future_LR', color='red')
plt.plot(xidx2d_future, y_future_predict_poly, label='Future_poly', color='red')
plt.plot(xidx2d_future, y_future_predict_avg, label='Future_avg', color='deeppink')

plt.plot(xidx2d, total, label='Real', color='blue', linewidth=1.0, linestyle='dotted')
plt.scatter(xidx2d, total, label='Real', color='red')

plt.title(f'{area}-模型预测', fontproperties=font)
plt.legend(loc='upper left')
plt.xticks(x,xlabel)

plt.savefig(picPath+'/'+fileName,bbox_inches='tight')
#plt.show()

params['diagramFileName']=fileName
print(json.dumps(params))
