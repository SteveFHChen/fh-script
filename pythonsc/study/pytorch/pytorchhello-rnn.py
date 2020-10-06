
#Sample command:
#conda activate pytorch
#python C:\fh\ws\ws1\fh-script\pythonsc\study\pytorch\pytorchhello-rnn.py

#Reference
#pytorch-RNN进行回归曲线预测
#任务：通过输入的sin曲线与预测出对应的cos曲线
#https://blog.csdn.net/maqunfi/article/details/84504645

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1) #为了可复现
 
#超参数设定
TIME_SETP=10
INPUT_SIZE=1
LR=0.02
DOWNLoad_MNIST=True

#定义RNN网络结构
from torch.autograd import Variable
class RNN(nn.Module):
    def __init__(self):
        #在这个函数中，两步走，先init,再逐步定义层结构
        super(RNN,self).__init__()
        
        self.rnn=nn.RNN(   #定义32隐层的rnn结构
          input_size=1,    
          hidden_size=32,  #隐层有32个记忆体
          num_layers=1,     #隐层层数是1
          batch_first=True 
        )
        
        self.out=nn.Linear(32,1)  #32个记忆体对应一个输出
    
    def forward(self,x,h_state):
        #前向过程，获取 rnn网络输出r_put(注意这里r_out并不是最后输出，最后要经过全连接层)  和  记忆体情况h_state
        r_out,h_state=self.rnn(x,h_state)        
        outs=[]#获取所有时间点下得到的预测值
        for time_step in range(r_out.size(1)): #将记忆rnn层的输出传到全连接层来得到最终输出。 这样每个输入对应一个输出，所以会有长度为10的输出
            outs.append(self.out(r_out[:,time_step,:]))
        return torch.stack(outs,dim=1),h_state  #将10个数 通过stack方式压缩在一起
 
rnn=RNN()
print('RNN的网络体系结构为：',rnn)

#创建数据集及网络训练
#定义优化器和 损失函数
optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_fun=nn.MSELoss()
h_state=None #记录的隐藏层状态，记住这就是记忆体，初始时候为空，之后每次后面的都会使用到前面的记忆，自动生成全0的
             #这样加入记忆信息后，每次都会在之前的记忆矩阵基础上再进行新的训练，初始是全0的形式。
#启动训练，这里假定训练的批次为100次
 
 
plt.ion() #可以设定持续不断的绘图，但是在这里看还是间断的，这是jupyter的问题
for step in range(100):
    #我们以一个π为一个时间步   定义数据，
    start,end=step*np.pi,(step+1)*np.pi
    
    steps=np.linspace(start,end,10,dtype=np.float32)  #注意这里的10并不是间隔为10，而是将数按范围分成10等分了
    
    x_np=np.sin(steps)
    y_np=np.cos(steps)
    #将numpy类型转成torch类型   *****当需要 求梯度时，一个 op 的两个输入都必须是要 Variable，输入的一定要variable包下
    x=Variable(torch.from_numpy(x_np[np.newaxis,:,np.newaxis]))#增加两个维度，是三维的数据。
    y=Variable(torch.from_numpy(y_np[np.newaxis,:,np.newaxis]))
    
    #将每个时间步上的10个值 输入到rnn获得结果     这里rnn会自动执行forward前向过程.  这里输入时10个，输出也是10个，传递的是一个长度为32的记忆体
    predition,h_state=rnn(x,h_state)
    
    #更新新的中间状态
    h_state=Variable(h_state.data)   #擦，这点一定要从新包装
    loss=loss_fun(predition,y)
    #print('loss:',loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    # plotting   画图，这里先平展了  flatten，这样就是得到一个数组，更加直接
    
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, predition.data.numpy().flatten(), 'b-')
    #plt.draw(); 
    plt.pause(0.05)
 
plt.ioff()  #关闭交互模式
plt.show()



