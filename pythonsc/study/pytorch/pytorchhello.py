
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

x=torch.linspace(-1,1,100).reshape(100,1)
#建立从-1到1之间一百个点并且改变他的阶数（从【1,100】变到【100,1】）
y=x.pow(2)
#建立x与y之间的关系y=x^2
y_real=torch.normal(y,0.05)
#在实际过程中由于不可避免的因素存在会有误差发生但是围绕实际值上下波动

class Net(torch.nn.Module):#继承Module这个类
    def __init__(self,n_input,n_hidden,n_output):#我们假设有三层分别是输入，隐藏，输出
        super(Net,self).__init__()#调用父类__init__()方法
        self.hidden=torch.nn.Linear(n_input,n_hidden)#搭建输入与隐藏层的关系
        self.predict=torch.nn.Linear(n_hidden,n_output)#搭建隐藏层与输出层的关系
        
    def forward(self,x):#这个方法使数据向前传递
        x=F.relu(self.hidden(x))#relu是激励函数，可以将直线弯曲（我们的二次曲线是弯的）
        x=self.predict(x)#将x从输出结果拿出来
        return x;#返回拟合结果x

net=Net(1,50,1)#我假设一个输入，隐藏层里有50个神经元，和一个输出
print('Net的网络体系结构为：', net)

optimizer=torch.optim.Adam(net.parameters(),0.01)#定义优化器
loss_func=torch.nn.MSELoss()#训练出来的结果和实际对比

for i in range(100):#我们训练一百次差不多了，如果要结果更加准确可以训练更多
    if i:
        loss.backward()#将误差返回给模型
        optimizer.step()#建模型的数据更新
    prediction=net(x)#将数据放入放入模型中
    loss=loss_func(prediction,y_real)#把模拟的结果和实际的差计算出来
    optimizer.zero_grad()
#将上一步计算的梯度清零，因为他计算的时候梯度会累加，这样会导致结果不准

plt.plot(x, y_real, color='blue', linewidth=1.0, linestyle='dotted')
plt.scatter(x, y_real, color="blue")
plt.plot(x.tolist(),prediction.tolist(),color="red")
plt.show()

