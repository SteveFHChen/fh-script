# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 15:45:03 2020

@author: steve
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

LEN = 80

#x_train_float=torch.linspace(-1,1,100).reshape(100,1,3)
x_train_float = torch.randn(1, 1, 96)
x_train_float = torch.tensor(np.arange(-8,8,1).reshape(2,1,8)).float() # 2批 x 1行 x 10列
x_train_float = torch.linspace(-15, 15, LEN).reshape(1, 1, -1).float()
#建立从-1到1之间一百个点并且改变他的阶数（从【1,100】变到【100,1】）
#y=x_train_float #使用一元一次函数检测卷积神经网络序列预测
y=x_train_float.pow(2) #使用一元二次函数检测卷积神经网络序列预测
#建立x与y之间的关系y=x^2
y_train_float=torch.normal(y, 5)
y_train_float=y
y_train_float_2d=y_train_float.view(1,-1)
#在实际过程中由于不可避免的因素存在会有误差发生但是围绕实际值上下波动

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
    

    
net = CNN_Series(LEN, 800, 16, LEN)
print('Net的网络体系结构为：', net)
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
loss_func = nn.MSELoss()

#Enable to train model if need

for i in range(880):#我们训练一百次差不多了，如果要结果更加准确可以训练更多
    for j in range(x_train_float.size(0)):
        x_train_floatx = x_train_float[j].reshape(1, 1, -1)
        y_train_floatx_2d=y_train_float[j].view(1,-1)
        if i:
            loss.backward()#将误差返回给模型
            optimizer.step()#建模型的数据更新
        prediction=net(x_train_floatx)#将数据放入放入模型中
        
        loss=loss_func(prediction,y_train_floatx_2d)#把模拟的结果和实际的差计算出来
        if i%10==0:
            print("i={}, Loss: {}".format( i, loss))
            #print("predition size:", prediction.size())
            #print("y_train_float_2d size:", y_train_float_2d.size())
        optimizer.zero_grad()

print("x_train_float.size: ", x_train_float.size())
print("y_train_float.size: ", y_train_float.size())
print("prediction.size: ", prediction.size())

#Save model
torch.save(net.state_dict(), "nn0.pkl")

#To improve performance, load existing model directly
net.load_state_dict(torch.load("nn0.pkl"))

from matplotlib.font_manager import FontProperties
font = {'family':'SimHei', 'weight':'normal', 'size':10} #字体设置方式2，可显示中文字体
plt.title(f"CNN Verification", fontproperties=font);

#Show real graph
plt.plot(x_train_float.view(-1).tolist(), y_train_float.view(-1).tolist(), color='blue', linewidth=1.0, linestyle='dotted')
plt.scatter(x_train_float.view(-1).tolist(), y_train_float.view(-1).tolist(), color="blue")

prediction = ''

#Show train graph
for j in range(x_train_float.size(0)):
    x_train_floatx = x_train_float[j].reshape(1, 1, -1)
    predictionx = net(x_train_floatx)
    if j==0:
        prediction = predictionx
    else:
        prediction = torch.cat([prediction, predictionx])

plt.plot(x_train_float.view(-1).tolist(), prediction.view(-1).tolist(),color="green", linewidth=2.0)

x_test_float = torch.linspace(-10, 25, LEN).reshape(1, 1, -1).float()
predictionx = net(x_test_float)
plt.plot(x_test_float.view(-1).tolist(), prediction.view(-1).tolist(),color="green", linewidth=2.0, linestyle='-.')

x_test_float = torch.linspace(-5, 30, LEN).reshape(1, 1, -1).float()
predictionx = net(x_test_float)
plt.plot(x_test_float.view(-1).tolist(), prediction.view(-1).tolist(),color="green", linewidth=2.0, linestyle='-.')

x_test_float = torch.linspace(0, 35, LEN).reshape(1, 1, -1).float()
predictionx = net(x_test_float)
plt.plot(x_test_float.view(-1).tolist(), prediction.view(-1).tolist(),color="green", linewidth=2.0, linestyle='-.')

'''
#Show test graph
x_test=np.arange(len(y_test)).reshape(-1,1)+len(x_train)+1
#x2=np.arange(500).reshape(-1,1)+len(x)+1
x_test_float=torch.from_numpy(x_test).float()
y_test_float=net(x_test_float)
plt.plot(x_test_float.tolist(), y_test_float.tolist(), color='green', linewidth=2.0)
#lt.scatter(x2float.tolist(), y2float.tolist(), color="red")
'''
'''
x_test_float = torch.tensor(np.array([
          -15, -12, -8, -7, -4, -3.5, -2.8, -0.3, #无序的
          0.3, 1.2, 2.5, 3.3, 7, 8, 11, 14, #无序的
          -3, -2, -1, 0, 1, 2, 3, 4, 
          7, 7.2, 7.8, 8.1, 8.3, 8.8, 9, 9.2, 
          9, 9.3, 9.5, 9.9, 10.11, 10.4, 10.7, 11.2, 
          14.3, 14.8, 15.1, 15.5, 15.8, 16, 16.2, 16.8, 
          16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5]).reshape(-1,1,8)).float()
#x_test_float = torch.linspace(-15, 15, 1*1*8).reshape(-1, 1, 8).float()
for j in range(x_test_float.size(0)):
    x_test_floatx = x_test_float[j].reshape(1, 1, -1)
    y_test_floatx = net(x_test_floatx)
    plt.plot(x_test_floatx.view(-1).tolist(), y_test_floatx.view(-1).tolist(), color='green', linewidth=2.0, linestyle='--')
'''

plt.legend()
plt.show()
print("End")