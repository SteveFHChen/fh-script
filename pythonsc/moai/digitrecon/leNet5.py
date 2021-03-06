
"""
    [pytorch]通过CNN实现手写数字识别（附完整代码）
    https://blog.csdn.net/qq_45402214/article/details/109989430?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control&dist_request_id=1328593.13860.16147804984764139&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets,transforms

#import sys
#sys.path.append('C:/fh/ws/ws1/fh-script/pythonsc/moai/digitrecon')

from mnistData import *
from fhDigitData import *

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2) #一张图使用6个5x5卷积核进行卷积，得到6个输出结果
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) #
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16*5*5,120)  # 第一层全连接，将16*5*5输入映射到120个输出
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) #经过3次全连接，最终得到10个输出，分别对应10个类别
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        in_size = x.size(0)
        out = self.relu(self.mp(self.conv1(x)))
        out = self.relu(self.mp(self.conv2(out)))
        #out = self.relu(self.conv3(out))
        out = out.view(in_size, -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        #使用多次全连接的目的就是为了增加网络的深度，使得网络更多参数对实际复杂情况更有适应性。
        return self.logsoftmax(out)
      
def trainWithMnistData(times=1):
    for epoch in range(times):
        model.train()
        for batch_index,(data,target) in enumerate(trainLoader):
            data,target = Variable(data), Variable(target)
            opt.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()#误差反向传播
            opt.step()#将参数更新值施加到net的parmeters上
            if batch_index%10 == 0:
                losses.append(loss.item())
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                   epoch, batch_index * len(data), len(trainLoader.dataset),
                   100. * batch_index / len(trainLoader), loss.item()))

def testWithMnistData():
    model.eval()
    test_loss = 0
    correct = 0
    for data,target in testLoader:
        data , target = Variable(data,volatile=True) ,Variable(target)
        output = model(data)
        test_loss += loss_func(output, target).item() # sum up batch loss
        pred = torch.max(output.data,1)[1]# get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(testLoader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testLoader.dataset),
        100. * correct / len(testLoader.dataset)))

#Use myself data to train
#试验说明对别分类训别问题的训练，打乱所有数据的顺序对训练非常重要。
def trainWithFhData(times=20):
    for epoch in range(times):
        model.train()
        #See any different with the below 2 training ways
        #Way 1: column by column
        #Fh测试过，这种方式训练的话Loss会收全敛的很好，最终搞啥别测试成功率可达95%。
        for c in range(8):
            for r in range(10):
        #Way 2: row by row
        #Fh测试过，这种方式训练的话Loss不会收敛，最终的识别测试时预测值一直会停留在某一两个值上，成功率也只有10%。
        #for r in range(10):
        #    for c in range(8):
                #r, c = 3, 5
                test_imx = getDigit(fhDigitImage, r, c)
                test_imx_torch = grayImage24DTensor(test_imx)
                
                opt.zero_grad() #这行在这里还是在最后都可以
                test_rate = model(Variable(test_imx_torch, volatile=True))
                #test_rate2 = Variable(test_rate, requires_grad=True)#这行不需要，否则Loss不收敛
                
                target = Variable(torch.LongTensor([r]))
                loss = loss_func(test_rate, target)
                #误差反向传播
                loss.backward()
                #将参数更新值施加到net的parmeters上
                opt.step()
            losses.append(loss.item())
            print(f"[ {epoch} - {c} ] Loss= {loss.item()}")

#Use myself data to test
def testWithFhData():
    successCount = 0
    test_columns = [2,4,6,7,8,9]
    for r in range(10):
        for c in test_columns:
            #r, c = 3, 5
            test_imx = getDigit(fhDigitImage, r, c)
            test_imx_torch = grayImage24DTensor(test_imx)

            test_rate = model(Variable(test_imx_torch, volatile=True))
            #test_rate = model(test_imx_torch)#Fh测试过，这样写也可以
            
            #方法1：Tensor转List，再取最大值的下标
            #rates = test_rate.view(-1).detach().numpy().tolist()
            #num_index = rates.index(max(rates))
            #方法2：直接取Tensor的最大值和对应下标
            num_index = torch.max(test_rate,1)[1]
            if r==num_index:
                successCount+=1
            #print(f"图片中的数字为 {r}, 识别结果为{num_index}，识别{'成功.' if r==num_index else '失败!'}")
            #plt.imshow(test_imx, plt.cm.gray)
    print(f"success rate: {successCount}/{10*len(test_columns)} = {round(successCount/10/len(test_columns)*100,2)}%")

def trainWithFhDataRandomRC(times=20):
    for epoch in range(times):
        model.train()
        for i, r in enumerate(rows[:100]):
            c = columns[i]
            test_imx = getDigit(fhDigitImage, r, c)
            test_imx_torch = grayImage24DTensor(test_imx)
            
            opt.zero_grad() #这行在这里还是在最后都可以
            test_rate = model(Variable(test_imx_torch, volatile=True))
            #test_rate2 = Variable(test_rate, requires_grad=True)#这行不需要，否则Loss不收敛
            
            target = Variable(torch.LongTensor([r]))
            loss = loss_func(test_rate, target)
            #误差反向传播
            loss.backward()
            #将参数更新值施加到net的parmeters上
            opt.step()
        losses.append(loss.item())
        print(f"[ {epoch} - {c} ] Loss= {loss.item()}")
            
            
def testWithFhDataRandomRC():
    testCount = 100
    successCount=0
    for i, r in enumerate(rows[120:]):
        c = columns[i]
        imx = getDigit(fhDigitImage, r, c)
        input = grayImage24DTensor(imx)
        #plt.imshow(imx, plt.cm.gray)
        
        #output = model(input)
        output = model(Variable(input, volatile=True))
        
        num_index = torch.max(output, 1)[1]
        if r==num_index:
            successCount+=1
        #print(f"Real:[{r}] - Predict:[{num_index}], {'成功.' if r==num_index else '失败!'}")
        #plt.pause(1)
    print(f"Random rows and columns, total success rate: {successCount}/{testCount}={round(successCount/testCount*100,2)}%")


model = LeNet5()
loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(),lr=0.001)

#误区1：分类识别问题的损失函数要用交叉熵函数CrossEntropyLoss()，而不是趋势预测的算术平方差MSELoss()，否则Loss无法收敛；
#误区2：分类识别问题的优化器必须使用torch.optim.Adam，而不使用趋势预测的torch.optim.SGD，否则Loss无法收敛；

losses = []

#随机产生元素为0-9的列表，从而保证训练数据是打乱顺序的
count = 200

rows = torch.rand(count)*10
rows=rows.int().numpy()

columns = torch.rand(count)*10
columns = columns.int().numpy()

if __name__ == "__main__":
    modelFile = "LeNet_MNIST.pkl"
    modelFile = "LeNet_FhDigit.pkl"
    
    #trainWithMnistData()
    #trainWithFhData()
    trainWithFhDataRandomRC()
    
    #保存训练模型
    #torch.save(model, modelFile)
    #model = torch.load(modelFile)
    
    #testWithMnistData()
    #testWithFhData()
    testWithFhDataRandomRC()

    #Plot losss change figure
    plt.plot(torch.arange(len(losses)).tolist(), losses, label='Loss')
    plt.legend(loc='upper left');
    plt.show()

"""
#Manual test
plt.imshow(testLoader.dataset[2][0][0], plt.cm.gray)

for data,target in testLoader:
    print(data.shape)
    

imgidx = 222
da = testLoader.dataset[imgidx][0][0]
inputs = torch.from_numpy(np.expand_dims(np.expand_dims(da, 0), 0)).float()

plt.imshow(da, plt.cm.gray)
torch.max(model(Variable(inputs, volatile=True)),1)[1]
"""