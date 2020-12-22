# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 20:58:47 2020

@author: steve
Refer:
    Pytorch中RNN和LSTM的简单应用 https://www.cnblogs.com/lokvahkoor/p/12263953.html
    使用RNN执行回归任务
    使用LSTM执行分类任务
    
标准和变种RNN、LSTM的论文理论讲解和原理图可从此网站取得：
    https://blog.csdn.net/zhaojc1995/article/details/80572098
    Pytorch实现LSTM和GRU https://blog.csdn.net/winycg/article/details/88937583
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 10      # rnn time step

#x_train_float=torch.linspace(-1,1,100).reshape(100,1,3)
x_train_float = torch.randn(1, 1, 96)
x_train_float = torch.tensor(np.arange(-8,8,1).reshape(2,1,8)).float() # 2批 x 1行 x 10列
x_train_float = torch.linspace(-10, 15, 10).reshape(1, -1, 1).float()
#建立从-1到1之间一百个点并且改变他的阶数（从【1,100】变到【100,1】）
#y=x_train_float #使用一元一次函数检测卷积神经网络序列预测
y=x_train_float.pow(2) #使用一元二次函数检测卷积神经网络序列预测
#建立x与y之间的关系y=x^2
y_train_float=torch.normal(y, 1)
#y_train_float=y
y_train_float_2d=y_train_float.view(1,-1)
#在实际过程中由于不可避免的因素存在会有误差发生但是围绕实际值上下波动


class RNN(nn.Module):
    def __init__(self, n_hidden1, n_layers):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=n_hidden1,     # rnn hidden unit
            num_layers=n_layers,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(n_hidden1, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # outs = outs.view(-1, TIME_STEP, 1)
        # return outs, h_state
        
        # or even simpler, since nn.Linear can accept inputs of any dimension 
        # and returns outputs with same dimension except for the last
        # outs = self.out(r_out)
        # return outs

net = RNN(32, 3)
print('Net的网络体系结构为：', net)
optimizer = torch.optim.Adam(net.parameters(), lr=0.2)   # optimize all cnn parameters
loss_func = nn.MSELoss()

h_state = None      # for initial hidden state
'''
for i in range(1000):
    if i:
        loss.backward()#将误差返回给模型
        optimizer.step()#建模型的数据更新
    prediction, h_state = net(x_train_float, h_state)#将数据放入放入模型中
    # !! next step is important !!
    h_state = h_state.data        # repack the hidden state, break the connection from last iteration
    
    loss=loss_func(prediction,y_train_float)#把模拟的结果和实际的差计算出来
    if i%200==0:
        print("i={}, Loss: {}".format( i, loss))
    optimizer.zero_grad()
#将上一步计算的梯度清零，因为他计算的时候梯度会累加，这样会导致结果不准

#Save model
torch.save(net.state_dict(), "rnn1.pkl")
'''
#To improve performance, load existing model directly
net.load_state_dict(torch.load("rnn1.pkl"))

#Show real graph
plt.plot(x_train_float.view(-1).tolist(), y_train_float.view(-1).tolist(), color='blue', linewidth=1.0, linestyle='dotted')
plt.scatter(x_train_float.view(-1).tolist(), y_train_float.view(-1).tolist(), color="blue")

print("h_state", h_state)
prediction, h_state = net(x_train_float, h_state)
print("h_state", h_state)

plt.plot(x_train_float.view(-1).tolist(), prediction.view(-1).tolist(),color="red", linewidth=2.0, linestyle='--')

x_test_float = torch.tensor(np.array([-19, -18, -17, -15, -13, -12, -8, -7, -4, -3.5, -2.8, -1.1, -0.3, 
          0.3, 1.2, 2.5, 3.3, 4, 7, 8, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 24]).reshape(1,-1,1)).float()
y_test_float, h_state = net(x_test_float, h_state)
plt.plot(x_test_float.view(-1).tolist(), y_test_float.view(-1).tolist(), color='green', linewidth=2.0)

plt.show()
