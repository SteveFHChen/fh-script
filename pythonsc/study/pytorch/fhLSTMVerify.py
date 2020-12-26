# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 20:45:55 2020

@author: steve
Refer:
    用 LSTM 做时间序列预测的一个小例子 https://www.jianshu.com/p/38df71cad1f6
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

#x_train_float=torch.linspace(-1,1,100).reshape(100,1,3)
x_train_float = torch.randn(1, 1, 96)
x_train_float = torch.tensor(np.arange(-8,8,1).reshape(2,1,8)).float() # 2批 x 1行 x 10列
x_train_float = torch.linspace(-15, 15, 10*1*8).reshape(1, -1, 1).float()
#建立从-1到1之间一百个点并且改变他的阶数（从【1,100】变到【100,1】）
#y=x_train_float #使用一元一次函数检测卷积神经网络序列预测
y=x_train_float.pow(2) #使用一元二次函数检测卷积神经网络序列预测
#建立x与y之间的关系y=x^2
y_train_float=torch.normal(y, 5)
y_train_float=y
y_train_float_2d=y_train_float.view(1,-1)
#在实际过程中由于不可避免的因素存在会有误差发生但是围绕实际值上下波动

y_train_float = torch.tensor(np.array([[ 15940],
       [ 15626],
       [ 20685],
       [ 25874],
       [ 14303],
       [ 12469],
       [ 18777],
       [ 18783],
       [ 20895],
       [ 31107],
       [ 23272],
       [ 25160],
       [ 26726],
       [ 21115],
       [ 29888],
       [ 27998],
       [ 28928],
       [ 23924],
       [ 27181],
       [ 34770],
       [ 26543],
       [ 41870],
       [ 34634],
       [ 19435],
       [ 50203],
       [ 21146],
       [ 44165],
       [ 48987],
       [ 42903],
       [ 60135],
       [ 59701],
       [ 41030],
       [ 41857],
       [ 63594],
       [ 49587],
       [ 21830],
       [ 89476],
       [ 59225],
       [ 30497],
       [ 50681],
       [ 70385],
       [ 57258],
       [ 62815],
       [ 36409],
       [ 71338],
       [ 69947],
       [ 69947],
       [ 78320],
       [ 49563],
       [ 59240],
       [ 74689],
       [ 74689],
       [103313],
       [ 50058],
       [ 47904],
       [ 81939],
       [ 59016],
       [ 52164],
       [112676],
       [ 41912],
       [ 69834],
       [ 62089],
       [ 74664],
       [ 91396],
       [ 76605],
       [ 22737],
       [106557],
       [ 65832],
       [ 66292],
       [ 94694],
       [ 93883],
       [ 88904],
       [ 99138],
       [ 67943],
       [ 76468],
       [ 76168],
       [104421],
       [ 76839],
       [111760],
       [ 82321],
       [100123],
       [ 89449],
       [ 85055],
       [ 97721],
       [ 81087],
       [ 86573],
       [ 95429],
       [ 93367],
       [100327],
       [100327],
       [ 89324],
       [ 96134],
       [ 77605],
       [ 72404],
       [125885],
       [ 45816],
       [ 69268],
       [ 88682],
       [ 77843],
       [115721],
       [ 70948],
       [ 63762],
       [ 76778],
       [ 73924],
       [ 40330],
       [ 77103],
       [104785],
       [ 51254],
       [ 66103],
       [ 96850],
       [ 45118],
       [ 69665],
       [ 73702],
       [ 67248],
       [ 59034],
       [ 61197],
       [ 37398],
       [ 70812],
       [ 56386],
       [ 25924],
       [ 54457],
       [ 47480],
       [ 44438],
       [ 44438],
       [ 56861],
       [ 31377],
       [ 64249],
       [ 24047],
       [ 84009],
       [ 30129],
       [ 42177],
       [ 54316],
       [ 47251],
       [ 28216],
       [ 58655],
       [ 39300],
       [ 52731],
       [ 45269],
       [ 25248],
       [ 57576],
       [ 31108]])).reshape(1, -1, 1).float()

x_train_float = torch.tensor(list(range(0, 141))).reshape(1, -1, 1).float()


class LSTM(nn.Module):
    def __init__(self,input_size=2,hidden_size=4,output_size=1,num_layer=2):
        super(LSTM,self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self,x):
        x, (h, c) = self.layer1(x, None)
        s,b,h = x.size()
        x = x.view(s*b,h)
        x = self.layer2(x)
        x = x.view(s,b,-1)
        return x
    
net = LSTM(1, 20, 1, 2)
print('Net的网络体系结构为：', net)
optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
loss_func = nn.MSELoss()

h_state = torch.randn(2, 114, 20)      # for initial hidden state
c_state = torch.randn(2, 114, 20)

for i in range(5000):
    if i:
        loss.backward()#将误差返回给模型
        optimizer.step()#建模型的数据更新
    #prediction, (h_state, c_state) = net(x_train_float, (h_state, c_state))#将数据放入放入模型中
    prediction = net(x_train_float)
    # !! next step is important !!
    #h_state = h_state.data        # repack the hidden state, break the connection from last iteration
    #c_state = c_state.date
    
    loss=loss_func(prediction,y_train_float)#把模拟的结果和实际的差计算出来
    if i%200==0:
        print("i={}, Loss: {}".format( i, loss))
    optimizer.zero_grad()
#将上一步计算的梯度清零，因为他计算的时候梯度会累加，这样会导致结果不准

#Save model
torch.save(net.state_dict(), "lstm2.pkl")

#To improve performance, load existing model directly
net.load_state_dict(torch.load("lstm2.pkl"))

#Show real graph
plt.plot(x_train_float.view(-1).tolist(), y_train_float.view(-1).tolist(), color='blue', linewidth=1.0, linestyle='dotted')
plt.scatter(x_train_float.view(-1).tolist(), y_train_float.view(-1).tolist(), color="blue")

print("h_state", h_state)
#prediction, (h_state, c_state) = net(x_train_float, (h_state, c_state))
prediction = net(x_train_float)
print("h_state", h_state)

plt.plot(x_train_float.view(-1).tolist(), prediction.view(-1).tolist(),color="red", linewidth=2.0)

x_test_float = torch.tensor(np.array([-19, -18, -17, -15, -13, -12, -8, -7, -4, -3.5, -2.8, -1.1, -0.3, 
          0.3, 1.2, 2.5, 3.3, 4, 7, 8, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 24]).reshape(1,-1,1)).float()
#y_test_float, (h_state, c_state) = net(x_test_float, (h_state, c_state))
y_test_float = net(x_test_float)
plt.plot(x_test_float.view(-1).tolist(), y_test_float.view(-1).tolist(), color='green', linewidth=2.0, linestyle='-.')

plt.show()