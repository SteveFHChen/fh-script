import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

from mnistData import *
from fhDigitData import *

#From: https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
class AlexNet1(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
#AlexNet MNIST Pytorch
#https://blog.csdn.net/xuan_liu123/article/details/89105997
class AlexNet2(nn.Module):
    def __init__(self):
        super(AlexNet2,self).__init__()

        # 由于MNIST为28x28， 而最初AlexNet的输入图片是227x227的。所以网络层数和参数需要调节
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) #AlexCONV1(3,96, k=11,s=4,p=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)#AlexPool1(k=3, s=2)
        self.relu1 = nn.ReLU()

        # self.conv2 = nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)#AlexCONV2(96, 256,k=5,s=1,p=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)#AlexPool2(k=3,s=2)
        self.relu2 = nn.ReLU()


        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)#AlexCONV3(256,384,k=3,s=1,p=1)
        # self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)#AlexCONV4(384, 384, k=3,s=1,p=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)#AlexCONV5(384, 256, k=3, s=1,p=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)#AlexPool3(k=3,s=2)
        self.relu3 = nn.ReLU()

        self.fc6 = nn.Linear(256*3*3, 1024)  #AlexFC6(256*6*6, 4096)
        self.fc7 = nn.Linear(1024, 512) #AlexFC6(4096,4096)
        self.fc8 = nn.Linear(512, 10)  #AlexFC6(4096,1000)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = x.view(-1, 256 * 3 * 3)#Alex: x = x.view(-1, 256*6*6)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        return x


def trainWithMnistData(times=1):
    for epoch in range(times):
        model.train()
        for batch_index,(data,target) in enumerate(trainLoader):
            data,target = Variable(data), Variable(target)
            opt.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            opt.step()
            if batch_index%20 == 0:
                 losses.append(loss.item())
                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                   epoch, batch_index * len(data), len(trainLoader.dataset),
                   100. * batch_index / len(trainLoader), loss.item()))

def testWithMnistData():
    #Fh测试过，AlexNet识别准确率可达96.79%，比LeNet的95%高一点点。
    with torch.no_grad():
        #在接下来的代码中，所有Tensor的requires_grad都会被设置为False
        correct = 0
        total = 0

        for data in testLoader:
            images, labels = data
            #images, labels = images.to(device), labels.to(device)

            out = model(images)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images:{}%'.format(100 * correct / total)) #输出识别准确率

#Use myself data to train model
def trainWithFhData(times=5):
    for epoch in range(times):
        model.train()
        for i, r in enumerate(rows):
            c = columns[i]
            input2 = grayImage24DTensor(getDigit(fhDigitImage, r, c))
            label2 = Variable(torch.LongTensor([r]))
            
            opt.zero_grad()
            output = model(Variable(input2, volatile=True))
            loss = loss_func(output, label2)
            loss.backward()
            opt.step()
            if i%10==0:
                losses.append(loss.item())
                print('[%d, %5d] loss:%.4f'%(epoch, (i)*epoch, loss.item()))
                
def trainWithFhData2(times=5):
    for epoch in range(times):
        model.train()
        for c in range(8):
            for r in range(10):
                #input2 = grayImage24DTensor(getDigit(fhDigitImage, r, c))
                test_imx = getDigit(fhDigitImage, r, c)
                input2 = grayImage24DTensor(test_imx)
                label2 = Variable(torch.LongTensor([r]))
                
                opt.zero_grad()
                output = model(Variable(input2, volatile=True))
                loss = loss_func(output, label2)
                loss.backward()
                opt.step()
            losses.append(loss.item())
            print('[%d, %5d] loss:%.4f'%(epoch, c, loss.item()))

def testWithFhData():
    testCount = 20
    successCount=0
    for i, r in enumerate(rows[:testCount]):
        c = columns[i]
        imx = getDigit(fhDigitImage, r, c)
        input = grayImage24DTensor(imx)
        #plt.imshow(imx, plt.cm.gray)
        
        #output = model(input)
        output = model(Variable(input, volatile=True))
        
        num_index = torch.max(output, 1)[1]
        if r==num_index:
            successCount+=1
        print(f"Real:[{r}] - Predict:[{num_index}], {'成功.' if r==num_index else '失败!'}")
        #plt.pause(1)
    print(f"Total success rate: {successCount}/{testCount}={round(successCount/testCount*100,2)}%")


#transform
"""
transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomGrayscale(),
                    transforms.ToTensor(),
])

transform1 = transforms.Compose([
                    transforms.ToTensor()
])
"""

# 加载数据
"""
trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True,num_workers=0)# windows下num_workers设置为0，不然有bug

testset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform1)
testloader = torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False,num_workers=0)
"""

model = AlexNet2()
loss_func = nn.CrossEntropyLoss()
#opt = optim.SGD(net.parameters(),lr=1e-4, momentum=0.9)
#opt = optim.SGD(net.parameters(),lr=0.001)
opt = torch.optim.Adam(model.parameters(),lr=0.001)
#device : GPU or CPU
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#net.to(device)
losses = []

#随机产生元素为0-9的列表，从而保证训练数据是打乱顺序的
count = 200

rows = torch.rand(count)*10
rows=rows.int().numpy()

columns = torch.rand(count)*10
columns = columns.int().numpy()

if __name__ == "__main__":
    modelFile = "AlexNet_MNIST.pkl"
    modelFile = "AlexNet_FhDigit.pkl"
    
    #trainWithMnistData()
    #trainWithFhData()
    trainWithFhData2()
    
    #保存训练模型
    torch.save(model, modelFile)
    model = torch.load(modelFile)
    
    #testWithMnistData()
    testWithFhData()

    #Plot losss change figure
    plt.plot(torch.arange(len(losses)).tolist(), losses, label='Loss')
    plt.legend(loc='upper left');
    plt.show()
    
"""
试验说明：
    AlexNet比LeNet5更深更复杂，从而要求有更多的训练数据和训练次数才能使得模型的Loss收敛。
    因此，需要根据实际情况来定神经网络的深度和复杂度。
"""

