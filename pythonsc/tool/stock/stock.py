# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:06:15 2020

@author: steve
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

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
def split_sequence(list, percent):
    a = int(len(list)*percent)
    x1 = list[:a]
    x2 = list[a:]
    return x1, x2

font = {'family':'SimHei', 'weight':'normal', 'size':10} #字体设置方式2，可显示中文字体

stocks = [
    {'No':0, 'code': '600036', 'name':'招商', 'trainTimes': 1000, 'lr': 0.0000155, 'split_ratio': 0.8 },
    {'No':1, 'code': '600519', 'name':'茅台', 'trainTimes': 1500, 'lr': 0.00000000988, 'split_ratio': 0.85 },
    {'No':2, 'code': '002460', 'name':'赣锋', 'trainTimes': 500, 'lr': 0.000001, 'split_ratio': 0.8 },
    {'No':3, 'code': '00700', 'name':'腾讯', 'trainTimes': 100, 'lr': 0.000000001, 'split_ratio': 0.8 },
    {'No':4, 'code': '03690', 'name':'美团', 'trainTimes': 100, 'lr': 0.000000001, 'split_ratio': 0.8 },
    {'No':5, 'code': 'PDD', 'name':'拼多多', 'trainTimes': 100, 'lr': 0.000000001, 'split_ratio': 0.8 },
    {'No':6, 'code': '000001', 'name':'上证指数', 'trainTimes': 100, 'lr': 0.000000001, 'split_ratio': 0.8 },
    {'No':7, 'code': '000300', 'name':'沪生300', 'trainTimes': 100, 'lr': 0.000000001, 'split_ratio': 0.8 },
    {'No':8, 'code': '399001', 'name':'深证成指', 'trainTimes': 100, 'lr': 0.000000001, 'split_ratio': 0.8 },
    {'No':9, 'code': '399005', 'name':'中小板指', 'trainTimes': 100, 'lr': 0.000000001, 'split_ratio': 0.8 },
    {'No':10, 'code': '399006', 'name':'创业板指', 'trainTimes': 100, 'lr': 0.000000001, 'split_ratio': 0.8 },
    {'code': '', 'name':'' },
    {'code': '', 'name':'' },
    {'code': '', 'name':'' },
    {'code': '', 'name':'' },
    {'code': '', 'name':'' }
    ]

plt.figure(figsize=(20,40),dpi=100)


see_stocks = [6, 7, 8, 9, 10, 0, 1, 2]
#see_stocks = [6, 7]
stocks_data = []
si=1
for s in see_stocks:
    stock = stocks[s]
    reader = csv.reader(open('../../data/stock-ss/'+stock['code']+'.csv'))
    data = []
    i=0
    for r in reader:
        #print(r)
        data.append(r)
        i+=1
        if i > 2000:
            break;
    del data[0]
    data.reverse() #倒序list内所有元素
    stocks_data.append(data)
    
    data_array = np.array(data)
    
    #plt.subplot(len(see_stocks),1,si);

    y_full = data_array[:, 6:7].astype(float).flatten()
    x_full = np.arange(len(data_array))
    plt.plot(x_full, y_full, label=stock['name'])
    
    y_full_max, y_full_min=y_full.max(), y_full.min()
    
    
    ds = data_array[:, 0]
    dslist = []
    for d in ds:
        dslist.append(d[5:])
    yearEnds=np.argwhere(np.array(dslist)=='12-27')
    
    for ye in yearEnds:
        plt.plot([ye[0], ye[0]], [y_full_min, y_full_max], linestyle='--')
        plt.text(ye[0], y_full_min, data_array[ye[0]][0])
        
    for yi in range(len(yearEnds)+1):
        #Find index of max and min in the year
        if yi==0:
            y_max_idx, y_min_idx = np.argmax(y_full[ : yearEnds[yi, 0]]), np.argmin(y_full[ : yearEnds[yi, 0]])
        elif yi==len(yearEnds):
            y_max_idx, y_min_idx = np.argmax(y_full[yearEnds[yi-1, 0] : ]) + yearEnds[yi-1, 0], np.argmin(y_full[yearEnds[yi-1, 0] : ]) + yearEnds[yi-1, 0]
        else:
            y_max_idx, y_min_idx = np.argmax(y_full[yearEnds[yi-1, 0] : yearEnds[yi, 0]]) + yearEnds[yi-1, 0], np.argmin(y_full[yearEnds[yi-1, 0] : yearEnds[yi, 0]]) + yearEnds[yi-1, 0]
        
        #Get value of max and min in the year
        y_max_val, y_min_val = y_full[y_max_idx], y_full[y_min_idx]
        full_delta = (y_full_max - y_full_min) * 0.1
        
        if y_min_idx <= y_max_idx:
            y_delta = round(y_max_val - y_min_val, 2)
        else:
            y_delta = -(y_max_val - y_min_val)
        y_per = round(y_delta / y_min_val * 100, 2)
        
        #Plot position of max and min in the year
        plt.plot([y_max_idx, y_max_idx], [y_max_val-full_delta, y_max_val+full_delta], color='red')
        plt.plot([y_min_idx, y_min_idx], [y_min_val-full_delta, y_min_val+full_delta], color='green')
        
        #Print max and min value, and the delta and percentage
        plt.text(y_max_idx, y_max_val+full_delta+20, data_array[y_max_idx][0])
        plt.text(y_max_idx, y_max_val+full_delta-20, y_max_val)
        
        plt.text(y_min_idx, y_min_val-full_delta+20, data_array[y_min_idx][0])
        plt.text(y_min_idx, y_min_val-full_delta-20, y_min_val)
        
        plt.text((y_max_idx - y_min_idx)/2+y_min_idx, y_min_val+full_delta+20, y_delta)
        plt.text((y_max_idx - y_min_idx)/2+y_min_idx, y_min_val+full_delta-20, str(y_per)+'%')
        
    
    plt.legend(loc='upper left', prop=font)
    si+=1


stock = stocks[10]
flag = ['model_CNN2x']
flag_train = ['model_CNN2']
'''
reader = csv.reader(open('../../data/stock-ss/'+stock['code']+'.csv'))
data = []
i=0
for r in reader:
    #print(r)
    data.append(r)
    i+=1
    if i > 2000:
        break;
del data[0]
data.reverse() #倒序list内所有元素

data_array = np.array(data)



y_full = data_array[:, 6:7].astype(float).flatten()
#x_full = len(data_array)-np.arange(len(data_array))
x_full = np.arange(len(data_array))

x_train, x_test=split_sequence(x_full, stock['split_ratio'])
y_train, y_test=split_sequence(y_full, stock['split_ratio'])

plt.plot([len(x_train), len(x_train)], [min(y_full), max(y_full)], linestyle='--')

x_train_float=torch.from_numpy(x_train).float()
y_train_float=torch.from_numpy(y_train).float()
x_test_float=torch.from_numpy(x_test).float()
y_test_float=torch.from_numpy(y_test).float()

x_train_float3d = x_train_float.reshape(1, -1, 1)
y_train_float3d = y_train_float.reshape(1, -1, 1)
x_test_float3d = x_test_float.reshape(1, -1, 1)

plt.plot(x_full, y_full, label='Open')
plt.plot(x_full, data_array[:, 3:4].astype(float).flatten(), label='Close')
#plt.plot(99-np.arange(100), data_array[0:100, 6:7].astype(float).flatten())
'''
'''
y1 = [4,5,6]
y2 = [2,3,3]
plt.bar([1,2,3], np.array(y1)-np.array(y2), bottom=np.array(y2), width=0.6)
'''



if 'model_CNN2' in flag:
    y_train_float1d = y_train_float.view(-1)
    tl = len(y_train_float1d) #Total length
    lx = 32 #Single batch length
    y_train2 = []
    for i in range(lx, tl):
        y_train2.append(np.array(y_train_float1d[i-lx : i]))
    y_train2_cnn = torch.tensor(y_train2).reshape(tl-lx, 1, lx).float()
    y_train2_cnc_real = torch.tensor(y_train_float1d[lx:]).reshape(-1, 1).float()
    x_train2_cnn = list(range(lx, tl))
    
    LEN = lx
    
    net = CNN_Series(LEN, 200, 32, 1)
    print('Net的网络体系结构为：', net)
    optimizer = torch.optim.SGD(net.parameters(), lr=stock['lr'])#India
    loss_func = nn.MSELoss()
    
    #Enable to train model if need
    if 'model_CNN2' in flag_train:
        for i in range(stock['trainTimes']):#我们训练一百次差不多了，如果要结果更加准确可以训练更多
            #for j in range(y_train2_cnn.size(0)):
            if i:
                loss.backward()#将误差返回给模型
                optimizer.step()#建模型的数据更新
            prediction=net(y_train2_cnn)#将数据放入放入模型中
            
            loss=loss_func(prediction,y_train2_cnc_real)#把模拟的结果和实际的差计算出来
            if i%20==0 and i>0:
                print("i={}, Loss: {}".format( i, loss))
                #print("predition size:", prediction.size())
                #print("y_train_float_2d size:", y_train_float_2d.size())
            optimizer.zero_grad()
        
        #Save model
        torch.save(net.state_dict(), 'stock-cnn2-'+stock['name']+'.pkl')
    
    #To improve performance, load existing model directly
    net.load_state_dict(torch.load('stock-cnn2-'+stock['name']+'.pkl'))
    
    prediction = None
    
    prediction=net(y_train2_cnn)
    plt.plot(x_train2_cnn, prediction.view(-1).tolist(), label='CNN回归', color="red", linewidth=2.0)
    
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
        
    plt.plot(list(range(tl, tl+len(prediction_test_list))), prediction_test_list, color="red", linewidth=2.0)



plt.legend(prop=font)
plt.show()

