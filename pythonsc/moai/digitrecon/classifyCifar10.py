"""
    计算机视觉入门--图像分类简介及算法
    https://blog.csdn.net/KobeLovesDawn/article/details/86771279
        Nearest Neighbor算法
        K-Nearest Neighbor算法
        Linear Classification算法
        CNN
"""
import torchvision
import torch
from torch import nn
from torch.autograd import Variable

from cifar10Data import *

class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*4*4, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.fc1(x.reshape(x.shape[0], -1))
        x = self.fc2(x)
        return x

def trainWithC10Data(times=1):
    for epoch in range(times):
        linear_classifier.train()
        total_loss = 0
        for batch_index, batch in enumerate(c10TrainLoader):
            #batch[0] = batch[0]
            batch[0].requires_grad = False
            scores = linear_classifier(batch[0])
            loss = loss_func(scores, batch[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        losses.append(total_loss.item())
        print(f'Batch[{epoch}] Total Loss =', total_loss.item())
        
def testWithC10Data():
    correct = 0
    for batch_index, batch in enumerate(c10TestLoader):
        scores = linear_classifier(batch[0])
        predict = torch.argmax(scores, dim=1)
        correct += torch.sum(predict == batch[1])
    correctRate = correct.item() / len(c10TestLoader.dataset) * 100
    accuracy.append(correctRate)
    print('Accuracy: %.2f' % (correctRate) + '%')
    
def verifySinglePicture(picIndex=1):
    #dataset是一个10000组的列表，每组数据为（图像数据，类别代码）
    datax = c10TestLoader.dataset[picIndex]
    imx, labelx = datax[0], datax[1]
    imx.requires_grad = False
    
    imx_4d = np.expand_dims(imx, 0) #(1, 3, 32, 32)
    imx_torch = torch.tensor(imx_4d).float()
    target = Variable(torch.LongTensor([labelx]))
    
    #rates = linear_classifier(Variable(imx_torch, volatile=True))#会有warning，写法过时了
    rates = linear_classifier(imx_torch)
    
    num_index = torch.max(rates, 1)[1]
    print(f"图片中的数字为 {target}, 识别结果为{num_index}，识别{'成功.' if target==num_index else '失败!'}")
    
    #如何把图像打印出来？？？
    plt.imshow(imx.reshape(32, 32, 3))
    #plt.imshow(imx_torch.transpose((0,2,3,1)))
    plt.show()
    
    
linear_classifier = LinearClassifier()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(linear_classifier.parameters(), lr=0.01)

losses = []
accuracy = []

if __name__ == '__main__':
    modelFile = "LinearClassifier_CIFAR10.pkl"
    #modelFile = "LinearClassifier_FhDigit.pkl"
    
    """
    for i in range(10):
        print(f"Cycle {i}")
        trainWithC10Data()
        testWithC10Data()
    """
    
    #保存训练模型
    #torch.save(linear_classifier, modelFile)
    linear_classifier = torch.load(modelFile)
    
    testWithC10Data()
    
    #Plot losss change figure
    plt.title(f'{modelFile}')
    plt.plot(torch.arange(len(losses)).tolist(), losses, label='Loss')
    plt.legend(loc='upper left');
    plt.show()
    
    #测试单张图像的识别
    verifySinglePicture(20)
    
    #手动执行，用于学习单张图片的处理过程
    """
    batch1 = list(iter(c10TestLoader))[1]
    batch1[0].shape #torch.Size([32, 3, 32, 32]) 表示32张 RGB 3通道 32px * 32px的图片
    batch1[0][3].shape #torch.Size([3, 32, 32]) 表示取第0个批次中第3张图片
    images, labels = batch1[0], batch1[1]
    images.requires_grad = False
    
    imx, labelx = images[3], labels[3]
    imx_4d = np.expand_dims(imx, 0) #(1, 3, 32, 32) 表示取第3张图片，并将数据转成4维，即当前批次只有1张RGB3通道图片
    imx_torch = torch.tensor(imx_4d).float()
    target = Variable(torch.LongTensor([labelx]))
    
    rates = linear_classifier(Variable(imx_torch, volatile=True))
    loss = loss_func(rates, target)
    """
