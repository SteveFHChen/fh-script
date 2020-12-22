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
    
net = LSTM(1, 20, 1, 2)
print('Net的网络体系结构为：', net)
optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
loss_func = nn.MSELoss()

h_state = None      # for initial hidden state
c_state = None

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
torch.save(net.state_dict(), "gru1.pkl")

#To improve performance, load existing model directly
net.load_state_dict(torch.load("gru1.pkl"))

#Show real graph
plt.plot(x_train_float.view(-1).tolist(), y_train_float.view(-1).tolist(), color='blue', linewidth=1.0, linestyle='dotted')
plt.scatter(x_train_float.view(-1).tolist(), y_train_float.view(-1).tolist(), color="blue")

print("h_state", h_state)
#prediction, (h_state, c_state) = net(x_train_float, (h_state, c_state))
prediction = net(x_train_float)
print("h_state", h_state)

plt.plot(x_train_float.view(-1).tolist(), prediction.view(-1).tolist(),color="red", linewidth=2.0, linestyle='--')

x_test_float = torch.tensor(np.array([-19, -18, -17, -15, -13, -12, -8, -7, -4, -3.5, -2.8, -1.1, -0.3, 
          0.3, 1.2, 2.5, 3.3, 4, 7, 8, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 24]).reshape(1,-1,1)).float()
#y_test_float, (h_state, c_state) = net(x_test_float, (h_state, c_state))
y_test_float = net(x_test_float)
plt.plot(x_test_float.view(-1).tolist(), y_test_float.view(-1).tolist(), color='green', linewidth=2.0)

plt.show()