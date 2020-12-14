# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:25:26 2020

@author: steve
"""

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

print("1D array...")
x1=torch.arange(1,7,1)
print(x1)

print(x1.T)

print("1D array end====================")

print("2D array...")
print("Reshape 1...")
x2=x1.reshape(-1,1) #n rows * 1 columns
print(x2)

print("2D array multipluy 1...")
# nx1 * 1xm = nxm
x22 = torch.arange(2)
print(x2*x22)

print("Reshape 2...")
x3=x1.reshape(3,2) #3 rows * 2 columns
print(x3)

print("2D array multipluy 3...")
# 3x2 * 2x3 = 3*3
x32 = torch.arange(1,7,1).reshape(2,3)
print(x3)
print(x32)
print(x3 @ x32)
print((x3 @ x32) * (x3 @x32))

print("2D array end====================")

print("3D array...")
x1=torch.arange(1,25,1) 
x4=x1.reshape(2,3,4) #2 channels * 3 rows * 4 columns
print(x4)
print(x4.size())

print(x4[0:1])
print(x4[0:1, 1:2, 2:3])
print("3D array end====================")

x=torch.ones((2,3,1))
print(x)

print("1D convolution...")
k1=torch.from_numpy(np.array([1,2,3]).reshape(1,1,3)).float()
t1=tuple([1])

m = nn.Conv1d(1, 1, 3, stride=1, padding=1) #接收1行数据，输出1行数据，核大小为3，步进为1，边沿各补1个
print(m)
print(m.weight) #查看卷积核
print(m.bias) #查看偏置系数

#input = torch.randn(5, 2, 3)
input = torch.arange(50).reshape(5,1,10) #5*2*3 #生成5通道1行10列的三维数据，即5个批次，每批为1行x10列
input = input.float() #To fix issue: Expected object of scalar type Long but got scalar type Float
print(input)
print(input.shape)

output = m(input) # out = sum(in * weight) + bias
print("Covolution result...")
print(output)
print(output.shape)
print("1D convolution end====================")

print("1D max pooling...")
p_max = nn.MaxPool1d(2)
print(p_max)
p_max_out = p_max(input) #Sampling: 5x1x10 will become 5x1x5, out = max(in each 2)
print(p_max_out)

print("1D avg pooling...")
p_avg = nn.AvgPool1d(2)
print(p_avg)
p_avg_out = p_avg(input)
print(p_avg_out)
print("1D pooling end====================")

print("1D sequential convolution and pooling...")
s_max = nn.Sequential(m, p_max)
print(s)
s_max_out = s_max(input)
print(s_max_out)
print(s_max_out.shape)
print("1D sequential convolution and pooling  end====================")

print("View...")
print(input)
print(input.size())

print("以(5, -1)即5行自行推断的列数将三维input的数据显示成二维数据")
input_view = input.view(input.size(0), -1)
print(input_view)

print("以(-1)即1行自行推断的列数将三维input的数据显示成一维数据")
input_view2 = input.view(input.size(0)*input.size(1)*input.size(2)) #方式1
input_view2 = input_view.view(-1) #方式2，等价
print(input_view2)
print("View end====================")

print("Permute...")


print("Full connect y=Ax+b ...")
print("一维全连接...")
fc = nn.Linear(in_features=50, out_features=2)
print(fc.weight)
print(fc.bias)
fc_out = fc(input_view2) # 1行x50列 -> 1行x2列
print(fc_out)

print("二维全连接...")
fc_2d = nn.Linear(in_features=10, out_features=2) #
print(fc_2d.weight)
print(fc_2d.bias)
fc_2d_out = fc_2d(input_view) # 5行x10列 -> 5行x2列
print(fc_2d_out)
print("Full connect end====================")

print("Loss function ...")
loss_func = nn.MSELoss() # sum( (xi-yi)^2 ) / n
x1 = torch.tensor(np.arange(3).reshape(1,1,3)).float()
x2 = torch.tensor(np.arange(3).reshape(1,1,3)).float()+2
loss = loss_func(x1, x2)
print(loss)
print("Loss function end====================")

