# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 12:35:23 2020

@author: steve
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 17:38:15 2020

@author: steve
"""


#Sample command:
#conda activate pytorch
#python C:\fh\ws\ws1\fh-script\pythonsc\study\pytorch\pytorchhello.py

#Reference
#使用PyTorch拟合曲线
#https://blog.csdn.net/qq_42024963/article/details/96423297

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from pymysql import *

class Net(nn.Module):#继承Module这个类
    def __init__(self,n_input,n_hidden,n_output):#我们假设有三层分别是输入，隐藏，输出
        super(Net,self).__init__()#调用父类__init__()方法
        self.hidden=nn.Linear(n_input,n_hidden)#搭建输入与隐藏层的关系
        self.predict=nn.Linear(n_hidden,n_output)#搭建隐藏层与输出层的关系
        
    def forward(self,x):#这个方法使数据向前传递
        x=F.relu(self.hidden(x))#relu是激励函数，可以将直线弯曲（我们的二次曲线是弯的）
        x=self.predict(x)#将x从输出结果拿出来
        return x;#返回拟合结果x

class LSTM(nn.Module):
    def __init__(self,input_size=2,hidden_size=4,output_size=1,num_layer=2):
        super(LSTM,self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self,x):
        x,_ = self.layer1(x)
        s,b,h = x.size()
        x = x.view(s*b,h)
        x = self.layer2(x)
        x = x.view(s,b,-1)
        return x

class GRU_Series(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=5, num_layers=1):
        super(GRU_Series, self).__init__()
        
        self.rnn = nn.GRU(
            input_size,     
            hidden_size,     # rnn hidden unit
            num_layers,       # 有几层 RNN layers
            batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
          
    def forward(self, x, h_state):
         r_out,h_state = self.rnn(x, h_state)
         out = self.out(r_out[:, -1, :])
         
         outs = []    # save all predictions
         for time_step in range(r_out.size(1)):    # calculate output for each time step
             outs.append(self.out(r_out[:, time_step, :]))
         return torch.stack(outs, dim=1), h_state

class CNN_Series(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output):
        super(CNN_Series, self).__init__()
        #Define the layers
        #Define 1st convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels = 1, #要求数据1行进入，可以多个列M，可以分N个通道即批次
                out_channels = n_hidden1, #输出为Nx8xM
                kernel_size = 3,
                padding = 1
                ),
            nn.ReLU(inplace=True), #激活函数，好像可有可无
            nn.MaxPool1d(kernel_size=2) #输入Nx8xM ->输出Nx8x(M/2)
            )
        #Define 2nd convolution layer
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels = n_hidden1, #要求数据8行进入，可以多个列M/2，可以分N个通道即批次
                out_channels = n_hidden2, #输出为Nx16x(M/2)
                kernel_size = 3,
                padding = 1
                ),
            nn.ReLU(inplace=True), #激活函数，好像可有可无
            nn.MaxPool1d(kernel_size=2) #输入Nx16x(M/2) ->输出Nx16x(M/2/2)
            )
        #Define full connect layer
        self.fc = nn.Linear(
            in_features = int(n_hidden2 * n_input / 4), #要求数据按N行 x in_features列的二维矩阵输入
            out_features = n_output #输出为N行 x 1列
            )
        
    def forward(self, indata):
        #Connect all user defined layers to make it work
        x = self.conv1(indata)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
    
from matplotlib.font_manager import FontProperties
mainPath = 'C:/fh/ws/ws1/fh-script/pythonsc/webcat/'
#font = FontProperties(fname=mainPath+"simsun.ttc", size=10) #字体设置方式1，可显示中文字体
font = {'family':'SimHei', 'weight':'normal', 'size':10} #字体设置方式2，可显示中文字体

from dbconnection import *
conn = connect(host=db_host, port=db_port, database=db_name,
               user=db_username,
               password=db_pwd, charset='utf8')

cs1 = conn.cursor()

#flag value: model_LR
flag = ['model_LRx', 'model_LR2x', 'model_LR3x', 'model_Netx', 'model_LSTMx', 'model_GRUx', 'model_CNNx', 'model_CNN2']
'''
    Net: looks very good
    LSTM: always return a line model
    GRU: train performance is very slow
'''
flag_train = ['model_Net','model_CNNx', 'model_CNN2x']
flag_legend_on = True
flag_legend = 0
legend_loc1 = 'upper left'
legend_loc2 = 'lower left'

area='玻利维亚';legend_loc=legend_loc2; area_code='BOL'; cnn2_train=550; cnn2_lr=0.000000007;
area='美国';    legend_loc=legend_loc1; area_code='USA'; cnn2_train=150; cnn2_lr=0.0000000000083;
#area='印度';   legend_loc=legend_loc1; area_code='IND'; cnn2_train=150; cnn2_lr=0.0000000000005;
#area='俄罗斯'; legend_loc=legend_loc1; area_code='RUS'; cnn2_train=150; cnn2_lr=0.0000000008;
#area='尼泊尔'; legend_loc=legend_loc1; area_code='NPL'; cnn2_train=150; cnn2_lr=0.00000000005;
#area='意大利'; legend_loc=legend_loc2; area_code='ITA'; cnn2_train=150; cnn2_lr=0.000000000008;

sql1 = f"""
	SELECT DATE_FORMAT(business_date, '%Y%m%d') business_date, new_confirmed, @rn:=@rn+1 rn,  DATE_FORMAT(business_date, '%m%d') biz_md
	FROM fact_covid t, (SELECT @rn:=0) r 
	WHERE AREA='{area}'
	ORDER BY business_date
"""
cs1.execute(sql1)
alldata0 = cs1.fetchall()
y_full = []
for s in alldata0:
    y_full.append(s[1])
y_full=np.array(y_full).reshape(-1,1)


x_full = np.arange(len(y_full)).reshape(-1,1)+1
y_train=y_full
y_train_float=torch.from_numpy(y_train).float()

sql_xticks = f"""
SELECT * FROM (
	SELECT DATE_FORMAT(business_date, '%Y%m%d') business_date, new_confirmed, @rn:=@rn+1 rn,  DATE_FORMAT(business_date, '%b-%d') biz_md
	FROM fact_covid t, (SELECT @rn:=0) r 
	WHERE AREA='玻利维亚'
	ORDER BY business_date
) t WHERE rn%22=0
"""
cs1.execute(sql_xticks)
alldata0 = cs1.fetchall()
xs = []
xticks = []
for s in alldata0:
    xs.append(s[2])
    xticks.append(s[3])
    
conn.close()

x_train=np.arange(len(y_train)).reshape(-1,1)+1
x_train_float=torch.from_numpy(x_train).float()


plt.title(f"{area}-日新增确诊人数", fontproperties=font);
plt.ylabel("日新增确诊人数", fontproperties=font)
plt.xlabel("日期", fontproperties=font)
plt.xticks(xs,xticks)

#Show real graph
plt.plot(x_full, y_full, color='blue', linewidth=1.0, linestyle='dotted')
#plt.scatter(x_full, y_full, label="Real data", color="blue")
plt.scatter(x_full, y_full, color="blue")

def split_sequence(list, percent):
    a = int(len(list)*percent)
    x1 = list[:a]
    x2 = list[a:]
    return x1, x2

x_train, x_test=split_sequence(x_full, 0.8)
y_train, y_test=split_sequence(y_full, 0.8)

x_train_float=torch.from_numpy(x_train).float()
y_train_float=torch.from_numpy(y_train).float()
x_test_float=torch.from_numpy(x_test).float()
y_test_float=torch.from_numpy(y_test).float()

x_train_float3d = x_train_float.reshape(1, -1, 1)
y_train_float3d = y_train_float.reshape(1, -1, 1)
x_test_float3d = x_test_float.reshape(1, -1, 1)

if 'model_LR' in flag:
    from sklearn import linear_model
    from sklearn.preprocessing import PolynomialFeatures
    
    #Model 1:
    model_LR = linear_model.LinearRegression()
    model_LR.fit(x_train, y_train)
    y_predict=model_LR.predict(x_test)
    #y_future_predict=model_LR.predict(xidx2d_future)
    x_future = np.arange(20).reshape(-1,1) + 1 + x_full.max()
    y_future_predict = model_LR.predict(x_future)
    
    plt.plot(x_train, model_LR.predict(x_train), label='一元一次线性回归训练', color='green')
    plt.plot(x_test, y_predict, label='一元一次线性回归预测', color='green', linestyle='-.')
    flag_legend += 1

if 'model_LR2' in flag:
    poly_reg = PolynomialFeatures(degree=2)
    x_train_poly = poly_reg.fit_transform(x_train)
    x_test_poly = poly_reg.fit_transform(x_test)
    #Model 2:
    model_LR_poly = linear_model.LinearRegression()
    model_LR_poly.fit(x_train_poly, y_train)
    y_predict_poly=model_LR_poly.predict(x_test_poly)
    
    plt.plot(x_train, model_LR_poly.predict(x_train_poly), label='一元二次非线性回归训练', color='darkorange')
    plt.plot(x_test, y_predict_poly, label='一元二次非线性回归预测', color='darkorange', linestyle='-.')
    #plt.plot(x_future, y_future_predict, label='LR', color='red')
    flag_legend += 1
    
if 'model_LR3' in flag:
    poly_reg3 = PolynomialFeatures(degree=3)
    x_train_poly3 = poly_reg3.fit_transform(x_train)
    x_test_poly3 = poly_reg3.fit_transform(x_test)
    #Model 2:
    model_LR_poly3 = linear_model.LinearRegression()
    model_LR_poly3.fit(x_train_poly3, y_train)
    y_predict_poly3=model_LR_poly3.predict(x_test_poly3)
    
    plt.plot(x_train, model_LR_poly3.predict(x_train_poly3), label='一元三次非线性回归训练', color='fuchsia')
    plt.plot(x_test, y_predict_poly3, label='一元三次非线性回归预测', color='fuchsia', linestyle='-.')
    #plt.plot(x_future, y_future_predict, label='LR', color='red')
    flag_legend += 1
    
if 'model_Net' in flag:
    net=Net(1,800, 1)#我假设一个输入，隐藏层里有50个神经元，和一个输出
    print('Net的网络体系结构为：', net)
    optimizer=torch.optim.Adam(net.parameters(),0.1)#定义优化器
    loss_func=torch.nn.MSELoss()#训练出来的结果和实际对比
    
    if 'model_Net' in flag_train:
        for i in range(2000):#我们训练一百次差不多了，如果要结果更加准确可以训练更多
            if i:
                loss.backward()#将误差返回给模型
                optimizer.step()#建模型的数据更新
            prediction=net(x_train_float)#将数据放入放入模型中
            loss=loss_func(prediction,y_train_float)#把模拟的结果和实际的差计算出来
            if i%200==0:
                print("i={}, Loss: {}".format( i, loss))
            optimizer.zero_grad()
        #将上一步计算的梯度清零，因为他计算的时候梯度会累加，这样会导致结果不准
        
        #Save model
        torch.save(net.state_dict(), "nn1.pkl")
    
    #To improve performance, load existing model directly
    net.load_state_dict(torch.load("nn1.pkl"))
    
    prediction=net(x_train_float)
    plt.plot(x_train_float.view(-1).tolist(), prediction.view(-1).tolist(), label='BPNN回归训练', color='black')
    #plt.plot(x_test, y_predict_poly3, label='Net回归预测', color='red', linestyle='-.')
    
    prediction=net(x_test_float)
    plt.plot(x_test_float.view(-1).tolist(), prediction.view(-1).tolist(), label='BPNN回归预测', color='black', linestyle='-.')
    
    flag_legend += 1
    
if 'model_LSTM' in flag:
    net = LSTM(1, 20, 1, 2)
    print('Net的网络体系结构为：', net)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.00005)
    loss_func = nn.MSELoss()
    
    h_state = None      # for initial hidden state
    c_state = None
    
    for i in range(10000):
        if i:
            loss.backward()#将误差返回给模型
            optimizer.step()#建模型的数据更新
        #prediction, (h_state, c_state) = net(x_train_float, (h_state, c_state))#将数据放入放入模型中
        prediction = net(x_train_float2)
        # !! next step is important !!
        #h_state = h_state.data        # repack the hidden state, break the connection from last iteration
        #c_state = c_state.date
        
        loss=loss_func(prediction,y_train_float2)#把模拟的结果和实际的差计算出来
        if i%200==0:
            print("i={}, Loss: {}".format( i, loss))
        optimizer.zero_grad()
        
    #Save model
    torch.save(net.state_dict(), "lstm1.pkl")
    
    #To improve performance, load existing model directly
    net.load_state_dict(torch.load("lstm1.pkl"))
    
    prediction = net(x_train_float3d)
    print("h_state", h_state)
    plt.plot(x_train_float3d.view(-1).tolist(), prediction.view(-1).tolist(), label='LSTM回归训练', color="green", linewidth=2.0)
    
    prediction = net(x_test_float3d)
    plt.plot(x_test_float3d.view(-1).tolist(), prediction.view(-1).tolist(), label='LSTM回归预测', color='green', linewidth=2.0, linestyle='-.')
    
    flag_legend += 1
    
if 'model_GRU' in flag:
    net = GRU_Series(1, 20, 1, 2)
    print('Net的网络体系结构为：', net)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    loss_func = nn.MSELoss()
    
    h_state = None      # for initial hidden state
    
    for i in range(500):
        if i:
            loss.backward()#将误差返回给模型
            optimizer.step()#建模型的数据更新
        prediction, h_state = net(x_train_float3d, h_state)#将数据放入放入模型中
        # !! next step is important !!
        h_state = h_state.data        # repack the hidden state, break the connection from last iteration
        
        loss=loss_func(prediction,y_train_float3d)#把模拟的结果和实际的差计算出来
        if i%200==0:
            print("i={}, Loss: {}".format( i, loss))
        optimizer.zero_grad()
    #将上一步计算的梯度清零，因为他计算的时候梯度会累加，这样会导致结果不准
    
    #Save model
    torch.save(net.state_dict(), "gru1.pkl")
    
    #To improve performance, load existing model directly
    net.load_state_dict(torch.load("gru1.pkl"))
    
    print("h_state", h_state)
    prediction, h_state = net(x_train_float3d, h_state)
    print("h_state", h_state)
    
    plt.plot(x_train_float3d.view(-1).tolist(), prediction.view(-1).tolist(),color="green", linewidth=2.0)
    
    y_test_float3d, h_state = net(x_test_float3d, h_state)
    plt.plot(x_test_float3d.view(-1).tolist(), y_test_float3d.view(-1).tolist(), color='green', linewidth=2.0, linestyle='-.')

    flag_legend += 1
    
if 'model_CNN' in flag:
    
    LEN = 120
    x_train_float110 = x_train_float.view(-1)[:LEN].reshape(1, 1, -1)
    y_train_float110 = y_train_float.view(-1)[:LEN].reshape(1, 1, -1)
    #LEN = len(x_train_float110.view(-1))
    
    net = CNN_Series(LEN, 800, 32, LEN)
    print('Net的网络体系结构为：', net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.000000005)
    loss_func = nn.MSELoss()
    
    #Enable to train model if need
    if 'model_CNN' in flag_train:
        for i in range(1000):#我们训练一百次差不多了，如果要结果更加准确可以训练更多
            for j in range(x_train_float110.size(0)):
                x_train_floatx = x_train_float110[j].reshape(1, 1, -1)
                y_train_floatx_2d=y_train_float110[j].view(1,-1)
                if i:
                    loss.backward()#将误差返回给模型
                    optimizer.step()#建模型的数据更新
                prediction=net(x_train_floatx)#将数据放入放入模型中
                
                loss=loss_func(prediction,y_train_floatx_2d)#把模拟的结果和实际的差计算出来
                if i%100==0:
                    print("i={}, Loss: {}".format( i, loss))
                    #print("predition size:", prediction.size())
                    #print("y_train_float_2d size:", y_train_float_2d.size())
                optimizer.zero_grad()
        
        #Save model
        torch.save(net.state_dict(), "cnn1.pkl")
    
    #To improve performance, load existing model directly
    net.load_state_dict(torch.load("cnn1.pkl"))
    
    prediction = None
    
    #Show train graph
    for j in range(x_train_float110.size(0)):
        x_train_floatx = x_train_float110[j].reshape(1, 1, -1)
        predictionx = net(x_train_floatx)
        if j==0:
            prediction = predictionx
        else:
            prediction = torch.cat([prediction, predictionx])
    
    plt.plot(x_train_float110.view(-1).tolist(), prediction.view(-1).tolist(), label='CNN回归训练', color="red", linewidth=2.0)
    
    x_test_float110 = torch.tensor(np.arange(50, 50+LEN, 1)).reshape(1, 1, -1).float()
    plt.plot(x_test_float110.view(-1).tolist(), prediction.view(-1).tolist(), label='CNN回归训练', color="red", linewidth=2.0, linestyle='-.')
    
if 'model_CNN2' in flag:
    
    y_train_float1d = y_train_float.view(-1)
    tl = len(y_train_float1d) #Total length
    lx = 64 #Single batch length
    y_train2 = []
    for i in range(lx, tl):
        y_train2.append(np.array(y_train_float1d[i-lx : i]))
    y_train2_cnn = torch.tensor(y_train2).reshape(tl-lx, 1, lx).float()
    y_train2_cnc_real = torch.tensor(y_train_float1d[lx:]).reshape(-1, 1).float()
    x_train2_cnn = list(range(lx, tl))
    
    LEN = lx
    
    net = CNN_Series(LEN, 80, 32, 1)
    print('Net的网络体系结构为：', net)
    optimizer = torch.optim.SGD(net.parameters(), lr=cnn2_lr)#India
    loss_func = nn.MSELoss()
    
    #Enable to train model if need
    if 'model_CNN2' in flag_train:
        for i in range(cnn2_train):#我们训练一百次差不多了，如果要结果更加准确可以训练更多
            #for j in range(y_train2_cnn.size(0)):
            if i:
                loss.backward()#将误差返回给模型
                optimizer.step()#建模型的数据更新
            prediction=net(y_train2_cnn)#将数据放入放入模型中
            
            loss=loss_func(prediction,y_train2_cnc_real)#把模拟的结果和实际的差计算出来
            if i%10==0 and i>0:
                print("i={}, Loss: {}".format( i, loss))
                #print("predition size:", prediction.size())
                #print("y_train_float_2d size:", y_train_float_2d.size())
            optimizer.zero_grad()
        
        #Save model
        torch.save(net.state_dict(), f'cnn2-{area_code}-{cnn2_lr}-{cnn2_train}.pkl')
    
    #To improve performance, load existing model directly
    net.load_state_dict(torch.load(f'cnn2-{area_code}-{cnn2_lr}-{cnn2_train}.pkl'))
    
    prediction = None
    
    prediction=net(y_train2_cnn)
    plt.plot(x_train2_cnn, prediction.view(-1).tolist(), label='CNN回归训练', color="red", linewidth=2.0)
    
    prediction_test_list = []
    
    #Based on known x, generate the 1st prediction y
    y_test2_cnn = torch.tensor(y_full[tl-lx : tl]).reshape(1, 1, -1).float()
    prediction = net(y_test2_cnn)
    #plt.scatter(list([tl-lx+64]), prediction.view(-1).tolist(), color="green")
    prediction_test_list.extend(prediction.view(-1).tolist())
    
    #Base on know x + predicted y, generate the next prediction y one by one
    for i in range(tl-lx+1, len(y_full)-lx):
        y_test2_cnn = torch.cat([y_test2_cnn.view(-1), prediction.view(-1)])[1:].reshape(1, 1, -1)
        prediction = net(y_test2_cnn)
        #plt.scatter(list([i+64]), prediction.view(-1).tolist(), color="green")
        prediction_test_list.extend(prediction.view(-1).tolist())
        
    plt.plot(list(range(tl, tl+len(prediction_test_list))), prediction_test_list, label='CNN回归测试', color="red", linewidth=2.0, linestyle='-.')
    
    flag_legend += 1
    
if flag_legend_on and flag_legend > 0:
    #plt.rcParams['font.sans-serif']=['Simhei']  #解决图例中文显示问题，目前只知道黑体可行
    #plt.legend(loc='lower left', prop=font)
    plt.legend(loc=legend_loc, prop=font);
    

plt.show()
print("End")
