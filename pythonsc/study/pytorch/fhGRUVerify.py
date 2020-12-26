# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 21:25:24 2020

@author: steve
Refer:
    吴恩达深度学习编程作业pytorch 版 gru时间序列 https://blog.csdn.net/weixin_41992565/article/details/91420086

"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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
    
net = GRU_Series(1, 30, 1, 3)
print('Net的网络体系结构为：', net)
optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
loss_func = nn.MSELoss()

h_state = None      # for initial hidden state

for i in range(5000):
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
torch.save(net.state_dict(), "gru1.pkl")

#To improve performance, load existing model directly
net.load_state_dict(torch.load("gru1.pkl"))

from matplotlib.font_manager import FontProperties
font = {'family':'SimHei', 'weight':'normal', 'size':10} #字体设置方式2，可显示中文字体
plt.title(f"GRU Verification", fontproperties=font);

#Show real graph
plt.plot(x_train_float.view(-1).tolist(), y_train_float.view(-1).tolist(), color='blue', linewidth=1.0, linestyle='dotted')
plt.scatter(x_train_float.view(-1).tolist(), y_train_float.view(-1).tolist(), color="blue", label="Real")

print("h_state", h_state)
prediction, h_state = net(x_train_float, h_state)
print("h_state", h_state)

plt.plot(x_train_float.view(-1).tolist(), prediction.view(-1).tolist(),color="green", label='Train', linewidth=2.0)

x_test_float = torch.tensor(np.array([-19, -18, -17, -15, -13, -12, -8, -7, -4, -3.5, -2.8, -1.1, -0.3, 
          0.3, 1.2, 2.5, 3.3, 4, 7, 8, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 24]).reshape(1,-1,1)).float()
y_test_float, h_state = net(x_test_float, h_state)
plt.plot(x_test_float.view(-1).tolist(), y_test_float.view(-1).tolist(), color='green', linewidth=2.0, label='Test', linestyle='-.')

plt.legend()
plt.show()

