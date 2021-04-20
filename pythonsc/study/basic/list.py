
#Study: += 和list.append()等价
dir([]) #查看List所拥有的方法
dir(()) #查看Tuple所拥有的方法

#List用+运算符代替append()，代码更简洁
a1 = []

a1.append('abc')
print(a1)

a1 += ['a', 1]
print(a1)

a1 += ['hello', True]
print(a1)


a1.append([4, 5])
a1 += [[6, 7]]
print(a1)

#Tuple用+运算符进行元素追加
(1, 2)+(3, 4)

a = (1, 2)
a += (3, 4)

#Python三目运算
a=1
b=2
r = 'yes' if a>b else 'no'

import json
with open("C:/fh/testenv1/moget/2212276.json") as f:
    labPic1=json.load(f)
    
with open("C:/fh/testenv1/moget/202720blb5jbc59x4vr9mh.json") as f:
    labPic1=json.load(f)

import skimage.io as io
imgFile = "C:/fh/testenv1/moget/page001-100_files/2212276.jpg"
imgFile = "C:/fh/testenv1/moget/webcat/img/sum_18-1_files/202720blb5jbc59x4vr9mh.jpg"
im_sk=io.imread(imgFile)

import matplotlib.pyplot as plt
plt.imshow(im_sk)
plt.pause(5)

#plt.imshow(im_sk[99:186, 36:138, :])

#以JSON格式存储的表格，进行根据列值使用where对行进行选择
#方式1：以数组List形式一次性返回所有结果
nais = [d for d in labPic1['shapes'] if d['label']=='nai']
#方式2：以记录形式返回，通过next()遍历，也可使用list()一次性获取并转化为List
nais2 = (d for d in labPic1['shapes'] if d['label']=='nai')
naisx = next(nais2)
list(nais2)

naix = nais[0]['points']
#plt.imshow(im_sk[int(naix[0][1]):int(naix[1][1]), int(naix[0][0]):int(naix[1][0])])

import numpy as np
naix = np.array(naix).astype(int)
wmin, wmax, hmin, hmax = naix[0][0], naix[1][0], naix[0][1], naix[1][1]
if wmin > wmax:
    temp = wmax
    wmax = wmin
    wmin = temp
if hmin > hmax:
    temp = hmax
    hmax = hmin
    hmin = temp
im_sk_nai = im_sk[hmin:hmax, wmin:wmax]
plt.imshow(im_sk_nai)

im_sk_nai.shape
#(87, 102, 3)

#图像拼接成一张大图
#1）水平拼接
naih2=np.hstack((im_sk_nai, im_sk_nai))
naih2.shape
#(87, 204, 3)
plt.imshow(naih2)
#2）竖直拼接
naiv2=np.vstack((im_sk_nai, im_sk_nai))
naiv2.shape
#(174, 102, 3)
plt.imshow()
#3）摆阵列
nai10x10 = None
for r in range(5):
    nair = im_sk_nai
    for c in range(5-1):
        nair = np.hstack((nair, im_sk_nai))
    if r==0:
        nai10x10 = nair
    else:
        nai10x10=np.vstack((nai10x10, nair))
plt.imshow(nai10x10)

#图像扩展、裁剪、缩放，以达到想要的尺寸
from PIL import Image
from PIL import Image
img1=Image.open(imgFile)
#缩放，不保持比例
img1.resize((100,100),Image.ANTIALIAS)
#缩放，保持比例
h,w = img1.size
rate = h/w
img1_100=img1.resize((100,int(100/rate)),Image.ANTIALIAS)
#缩放，保持比例，并要求图像大小，不足就补白边, 以使图片大小符合神经网络的输入规格
background = Image.new('RGBA', (200, 200), (255, 255, 255, 255))
background.paste(img1_100, (10, 20))

#大小不一的图，先转成大小一致再拼接

plt.imshow(im_sk*[1, 0, 0]) #To confirm the 1st dimension is red
plt.imshow(im_sk*[0, 1, 0]) #To confirm the 2nd dimension is green
plt.imshow(im_sk*[0, 0, 1]) #To confirm the 3nd dimension is blue

plt.imshow(im_sk@[0.33, 0.33, 0.33], plt.cm.gray) 
#To distigush * and @, * keep shape, no add operation, @ has add operation, will narrow down dimension.

plt.imshow(im_sk[0:150, 0:100]) 
#To understand start point and direction at picture, also the mapping to 3D array.
#So we can know: 
#   the start point is at top left, and mapping [0, 0, :]
#   height(vertical) is the 1st dimension
#   width(horizental) is the 2nd dimension
#   deep is the 3nd dimension, in picture is called channel.


import numpy as np
a1 = np.arange(6)
a2 = np.arange(6).reshape(2,3)
a3 = np.arange(24).reshape(2,3,4)
a3pic = a3.transpose(0, 2, 1)

#np的卷积运算函数
result = np.convolve(a1, kernel)

#自实现优化的平滑卷积运算 —— 借助numpy高效的矩阵运算，省去使用低效率的双for循环
#步进1，形状不变
#Kernal: 1x3      (1x6x3) x (3x1) = 1x6
kernel = np.array([-1, 0, 1])
layer1 = np.hstack(([0], a1))[:len(a1)]
layer3 = np.hstack((a1, [0]))[-len(a1):]
compose = np.dstack( (np.dstack((layer1, a1)), layer3))
result = np.squeeze(compose@kernel[::-1]) 
#squeeze()和expand_dim()相么，分别是降维和增维
#[::-1]表示反转数组, -1表示反向步进，值为1

#Kernal: 2x2
#二维类似一维，只是要在每个通道上分别做。
#kernal: 3x3
#三维类似一维，只是要在每个通道上分别做。

#数组镜像
#1D上下镜像（照镜子梳头）
a1[::-1]

#2D上下镜像（湖面倒映）
a2[::-1]
#2D左右镜像（照镜子梳头）
a2.T[::-1].T

#3D -> RGB图上下镜像（湖面倒映）
a3pic[::-1]
#3D -> RGB图水平镜像（照镜子梳头）
#代码1
a3pic.transpose(1,0,2)[::-1].transpose(1,0,2)
#代码2
result = np.zeros(a3pic.shape)
for i in range(a3pic.shape[0]):
    result[i] = a3pic[i][::-1]

#3D 每个最小数组反转 -> RGB <-->BGR，R与B对调
a3pic.T[::-1].T

#图像缩放
#方法1：使用图像处理库自带函数
#   cv2.resize(img,(new_w, new_h))
#   sk_im.resize()试用成功，rescale没试用成功，通道会少了一维
from skimage.transform import resize,rescale
#按像素缩放
plt.imshow(resize(im_sk, (300, 200)))#不保持比例，注意顺序(h, w)
plt.imshow(resize(im_sk, (300, int(im_sk.shape[1]/im_sk.shape[0]*300))))#保持比例
#按百分比缩放
plt.imshow(resize(im_sk, (int(im_sk.shape[0]*0.8), int(im_sk.shape[1]*0.6))))#不保持比例
plt.imshow(resize(im_sk, (int(im_sk.shape[0]*0.7), int(im_sk.shape[1]*0.7))))#保持比例

#方法2：自定义缩放函数
#Fh暂时不花时间在这个的实现上

#保持比例补黑边
#如：要求统一出512x512图片作为神经网络训练的数据源
#做法：先将就大边调为512，在大边的外侧加对称黑边
#再封装成函数
if im_sk.shape[0] >= im_sk.shape[1]:
    new_h = 512
    new_w = int(im_sk.shape[1]/im_sk.shape[0]*512)
else:
    new_w = 512
    new_h = int(im_sk.shape[0]/im_sk.shape[1]*512)
delta = (512 - new_h, 512 - new_w)
if delta[0]:#高度不够，上下补边
    plt.imshow(
        np.vstack((
            np.vstack((#上边补边
                np.zeros((int(delta[0]/2), new_w, 3), np.uint8),
                resize(im_sk, (new_h, new_w))
                )),
            np.zeros((int(delta[0]/2), new_w, 3), np.uint8)#下边补边
            ))
        )
elif delta[1]:#宽度不够，左右补边
    plt.imshow(
        np.hstack((
            np.hstack((#左边补边
                np.zeros((new_h, int(delta[1]/2), 3), np.uint8),
                resize(im_sk, (new_h, new_w))
                )),
            np.zeros((new_h, int(delta[1]/2), 3), np.uint8)#右边补边
            ))
        )
else:#两边刚好够
    plt.imshow(resize(im_sk, (new_h, new_w)))


    
im_sk_v = im_sk.transpose(1,0,2)
result = fhresize(im_sk_v, (512, 512))
#result = fhresize(im_sk, (512, 512))
plt.imshow(
    np.vstack((
        np.hstack((result[0], result[1])),
        np.hstack((result[2], result[3]))
        ))
    )


#Labelme
#1）不保存图片数据--nodata
#2）自动保存 --autosave，JSON文件名与图片文件名一致
#3）自动检测标注
#4）批量标注
(fhlabelme) C:\Users\steve>labelme dog.jpg --labels labels.txt --nodata --autosave

#读取所有labelme json获取图像拼成大图展示
import os
fileList = os.listdir("C:/fh/testenv1/moget/shot1")
fileList = [d for d in fileList if d.endswith(".json")]


#图像增强
#颜色空间转换RGB<-->HSV
import skimage
im_sk_hsv = skimage.color.rgb2hsv(im_sk) #rgb -> hsv
plt.imshow(im_sk_hsv)
plt.imshow(im_sk_hsv, plt.cm.hsv)
plt.imshow(skimage.color.hsv2rgb(im_sk_hsv))#hsv -> rgb


"""
    图像颜色增强算法Opencv
    https://blog.csdn.net/hyqwmxsh/article/details/77980709
    
    数据增强-亮度-对比度-色彩饱和度-色调-锐度 不改变图像大小
    https://my.oschina.net/u/4341020/blog/3482780
    
    图像色彩增强之python实现——MSR,MSRCR,MSRCP,autoMSRCR
    https://blog.csdn.net/weixin_38285131/article/details/88097771
    
"""

#图像处理实操==========>
#梳头镜像旧代码
#im_sk_mirror = np.zeros(im_sk.shape).astype(np.uint8)#生成形状一样的空数组
##im_sk_mirror = im_sk.copy()#通过拷贝的方式获得一个形状一样的数组
#for i in range(im_sk.shape[0]):
#    im_sk_mirror[i] = im_sk[i][::-1]
#plt.imshow(im_sk_mirror)

#梳头镜像改进代码
im_sk_mirror = im_sk.transpose(1,0,2)[::-1].transpose(1,0,2)

plt.figure(figsize=(9,6), dpi=100) 

np.hstack((im_sk, im_sk_mirror)).shape
#照镜子梳头镜像
plt.imshow(np.hstack((im_sk, im_sk_mirror)))
#四方位镜像
#方式1：先照镜子梳头镜像，再对它像一起镜像一次（简单易理解）
plt.imshow(
    np.vstack((
        np.hstack((im_sk, im_sk_mirror)),
        np.hstack((im_sk, im_sk_mirror))[::-1]
    ))
    )
#方式2：先基于原图照镜子梳头镜像生成上图，再基于原图湖面倒映再照镜子梳头镜像生成下图，再合并成大图
im_sk_lake_mirror = np.zeros(im_sk.shape).astype(np.uint8)
for i in range(im_sk.shape[0]):
    im_sk_lake_mirror[i] = im_sk[::-1][i][::-1]
    
plt.imshow(
    np.vstack((
        np.hstack((im_sk, im_sk_mirror)),
        np.hstack((im_sk[::-1], im_sk_lake_mirror))
        ))
    )
    
#Fh总结：
#1. 图像方位变换8图（旋转、镜像）
#横向
plt.imshow(
    np.vstack((
        np.hstack((
            im_sk,
            im_sk.transpose(1,0,2)[::-1].transpose(1,0,2)
            )),
        np.hstack((
            im_sk[::-1],
            im_sk.transpose(1,0,2)[::-1].transpose(1,0,2)[::-1]
            ))
        ))
    )
#竖向
plt.imshow(
    np.vstack((
        np.hstack((
            im_sk.transpose(1,0,2),
            im_sk[::-1].transpose(1,0,2)
            )),
        np.hstack((
            im_sk.transpose(1,0,2)[::-1],
            im_sk[::-1].transpose(1,0,2)[::-1]
            ))
        ))
    )
#2. 图像通道变换8图（彩色、黑白）
#彩色
plt.imshow(
    np.vstack((
        np.hstack((
            im_sk, #RGB
            im_sk*[1, 0, 0] #R
            )),
        np.hstack((
            im_sk*[0, 1, 0],#G
            im_sk*[0, 0, 1] #B
            ))
        ))
    )
#通道任意排序
plt.imshow(
    np.vstack((#大图
        np.hstack((#上层图
            np.hstack((#上层图的前2幅
                im_sk,#RGB
                np.dstack((im_sk[:,:,1:2], np.dstack((im_sk[:,:,0:1], im_sk[:,:,2:3]))))#GRB
                )),
            np.dstack((im_sk[:,:, 2:3], im_sk[:,:, 0:2])) #BRG
            )),
        np.hstack((#下层图
            np.hstack((#下层图的前2幅
                np.dstack((im_sk[:,:,0:1], np.dstack((im_sk[:,:,2:3], im_sk[:,:,1:2])))),#RBG
                np.dstack((im_sk[:,:,1:2], np.dstack((im_sk[:,:,2:3], im_sk[:,:,0:1])))) #GBR
                )),
            np.dstack((im_sk[:,:,2:3], np.dstack((im_sk[:,:,1:2], im_sk[:,:,0:1])))) #BGR
            #等价于im_sk.transpose(2,0,1)[::-1].transpose(1,2,0)#BGR
            
            ))
        ))
    )
#由于[::-1]只能进行排序，不能任意排序，所以无法实现各种RGB通道排序

#黑白
plt.imshow(
    np.vstack((
        np.hstack((
            im_sk@[0.3, 0.3, 0.3], #RGB
            im_sk@[1, 0, 0] #R
            )),
        np.hstack((
            im_sk@[0, 1, 0],#G
            im_sk@[0, 0, 1] #B
            ))
        ))
    , plt.cm.gray
    )
    
    
#修改文件名
#OpenCV加载图片，文件名不能有空格，也不能有中文
#SKImage加载图片，文件名不能有空格，但可以有中文
#cd C:\fh\testenv1\moget\shot2
import os
images="."
oriNames = os.listdir(images)
newNames = []
for x in oriNames:
    if len(x.split(" "))>=2:
        newNames += [x.split(" ")[0]+"-"+x[-9:]]
    else:
        newNames += [x]
for i in range(1, len(oriNames)):
    os.rename(oriNames[i], newNames[i])
	
newNames = [x.split(" ")[0]+"-"+x[-9:] for x in oriNames if len(x.split(" "))>=2]
map(os.rename, oriNames, newNames)
    
