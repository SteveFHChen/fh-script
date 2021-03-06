
"""
    MNIST手写数字数据集读取方法
    https://blog.csdn.net/panrenlong/article/details/81736754
    
        训练样本：共60000个，其中55000个用于训练，另外5000个用于验证
        测试样本：共10000个
        3、数据集中像素值
        a）使用python读取二进制文件方法读取mnist数据集，则读进来的图像像素值为0-255之间；标签是0-9的数值。
        b）采用TensorFlow的封装的函数读取mnist，则读进来的图像像素值为0-1之间；标签是0-1值组成的大小为1*10的行向量。

"""
import numpy as np
import struct
import matplotlib.pyplot as plt

import os
import gzip

from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,transforms

mnistFiles = {
    "path": "C:/fh/testenv1/dataset/mnist",
    
    "train_images_file": "train-images.idx3-ubyte",
    "train_labels_file": "train-labels.idx1-ubyte",
    "test_images_file": "t10k-images.idx3-ubyte",
    "test_labels_file": "t10k-labels.idx1-ubyte",
    
    "train_images_gz_file": "train-images-idx3-ubyte.gz",
    "train_labels_gz_file": "train-labels-idx1-ubyte.gz",
    "test_images_gz_file": "t10k-images-idx3-ubyte.gz",
    "test_labels_gz_file": "t10k-labels-idx1-ubyte.gz"
}

for key in mnistFiles.keys():
    if key != "path":
        mnistFiles[key]=os.path.join(mnistFiles['path'], mnistFiles[key])

def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    print(offset)
    fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    #plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        #print(images[i])
        offset += struct.calcsize(fmt_image)
#        plt.imshow(images[i],'gray')
#        plt.pause(0.00001)
#        plt.show()
    #plt.show()

    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def load_train_images(idx_ubyte_file=mnistFiles["train_images_file"]):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)

def load_train_labels(idx_ubyte_file=mnistFiles["train_labels_file"]):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)

def load_test_images(idx_ubyte_file=mnistFiles["test_images_file"]):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)

def load_test_labels(idx_ubyte_file=mnistFiles["test_labels_file"]):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)

def loadMnistDataWay1():
    trainImages = load_train_images()
    trainLables = load_train_labels()
    testImages = load_test_images()
    testLables = load_test_labels()

    # 查看前十个数据及其标签以读取是否正确
    for i in range(10):
        print(trainLables[i])
        plt.imshow(trainImages[i], cmap='gray')
        plt.pause(0.000001)
        plt.show()
    print('done')
    
    return (trainImages, trainLables), (testImages, testLables)
#(trainImages, trainLables), (testImages, testLables) = loadMnistDataWay1()


# 定义加载数据的函数，data_folder为保存gz数据的文件夹，该文件夹下有4个文件
# 'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
# 't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'

def loadMnistDataWay2():
    with gzip.open(mnistFiles["train_labels_gz_file"], 'rb') as lbpath:
        trainLables = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(mnistFiles["train_images_gz_file"], 'rb') as imgpath:
        trainImages = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(trainLables), 28, 28)

    with gzip.open(mnistFiles["test_labels_gz_file"], 'rb') as lbpath:
        testLables = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(mnistFiles["test_images_gz_file"], 'rb') as imgpath:
        testImages = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(testLables), 28, 28)

    return (trainImages, trainLables), (testImages, testLables)
#(trainImages, trainLables), (testImages, testLables) = loadMnistDataWay2()


#Transfer MNIST file data to dataset
class MnistDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.trainImages = images
        self.trainLables = labels
        self.transform = transform
    
    def __getitem__(self, index):
        img, target = self.trainImages[index], int(self.trainLables[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.trainImages)

(trainImages, trainLables), (testImages, testLables) = loadMnistDataWay2()

trainDataset = MnistDataset(trainImages, trainLables, transform=transforms.ToTensor())
testDataset = MnistDataset(testImages, testLables, transform=transforms.ToTensor())

# 训练数据和测试数据的装载
trainLoader = DataLoader(
    dataset=trainDataset,
    batch_size=128, # 一个批次可以认为是一个包，每个包中含有128张图片
    shuffle=False
)

testLoader = DataLoader(
    dataset=testDataset,
    batch_size=100,
    shuffle=False
)

#Understand the data structure
"""
list(enumerate(trainLoader))[0]
type(list(enumerate(trainLoader))[1])

enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。

for batch_index,(data,target) in enumerate(train_loader):
    print(data.shape)
    print(target.shape)
    data,target = Variable(data), Variable(target)
"""
