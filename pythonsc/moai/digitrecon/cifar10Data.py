"""
    读取cifar数据集
    https://blog.csdn.net/daoyone/article/details/100547068?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1.control&dist_request_id=&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1.control
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader,Dataset
import torchvision
#from torchvision import datasets,transforms

cifar10Files = {
	"path": "C:/fh/testenv1/dataset/cifar-10-batches-py",
	
	"b1": "data_batch_1",
	"b2": "data_batch_2",
	"b3": "data_batch_3",
	"b4": "data_batch_4",
	"b5": "data_batch_5",
	"t1": "test_batch"
}

for key in cifar10Files.keys():
    if key != "path":
        cifar10Files[key]=os.path.join(cifar10Files['path'], cifar10Files[key])

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
def loadCifar10Data():
    b1 = unpickle(cifar10Files["b1"])
    return b1

def format2Images(bx):
    new = bx[b'data'].reshape(10000,3,32,32)
    #将[10000][3][32][32]转为[10000][32][32][3]
    imgs = new.transpose((0,2,3,1))
    return imgs

"""
batches.meta 文件
    num_cases_per_batch: 10000
    label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_vis: 3072
    
data_batch_x 和test_batch文件：
    b1.keys()
        dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
        batch_label: 对应的值是一个字符串，用来表明当前文件的一些基本信息。
        labels: 对应的值是一个长度为10000的列表，每个数字取值范围 0~9，代表当前图片所属类别
        data: 10000 * 3072 的二维数组，每一行代表一张图片的像素值。（32*32*3=3072）
        filenames: 长度为10000的列表，里面每一项是代表图片文件名的字符串。
    
    b1[b'batch_label']
    
    type(b1[b'labels'])
    len(b1[b'labels']) #10000
    b1[b'labels'][:10]
    
    type(b1[b'data'])
    b1[b'data'].shape #(10000, 3072), 3072 = 32px * 32px * 3channel
    img = b1[b'data'][0].reshape(3, 32, 32)
    
    len(b1[b'filenames']) #10000
    b1[b'filenames'][:10]
"""


"""
    pytorch如何导入本地数据集（CIFAR10为例）——详细教程
    使用下载好的本地CIFAR10数据建立Dataset
    https://zhuanlan.zhihu.com/p/129081723
    https://blog.csdn.net/weixin_44844089/article/details/106839856
    
    使用sys.path查看python的库目录，然后去对应的目录下找到cifar.py
    C:\fh\pf2\anaconda3\envs\pytorch\Lib\site-packages\torchvision\datasets\cifar.py
    修改里面的url即可。
    url = "file:///C:/fh/testenv1/dataset/cifar-10-python.tar.gz"
"""
c10TrainDS = torchvision.datasets.CIFAR10(
    root='C:/fh/testenv1/dataset/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

c10TestDS = torchvision.datasets.CIFAR10(
    root='C:/fh/testenv1/dataset/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

c10TrainLoader = DataLoader(c10TrainDS, batch_size=32, shuffle=False)
c10TestLoader = DataLoader(c10TestDS, batch_size=32, shuffle=False)

"""
    len(list(enumerate(c10TrainLoader)))
    len(list(iter(c10TrainLoader)))
"""
