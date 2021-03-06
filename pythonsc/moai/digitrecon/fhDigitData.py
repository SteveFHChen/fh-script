import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import datetime

import torch

im = np.zeros((10, 10))
im = np.arange(100).reshape(10, -1)

imgWidth = 1+28+1 #1个像素边框，28个内容像素
imgCount = 10

def generateHandWrittenDigit(imFile):
    #数字图片大小28p*28p, 大图中数字图片个数10个*10个,每个数字图片有1px的像素边框
    im = np.ones((imgWidth*imgCount, imgWidth*imgCount)) * 255
    im.shape

    plt.imshow(im, plt.cm.gray)

    #range(2,4)和np.arange(2,4)等价，只是写法和用的库不一样
    for i in range(imgCount):
        #Draw horizental line
        im[30*i  : 30*i+1] = 0
        im[30*(i+1)-1: 30*(i+1)] = 0
        #Draw vertical line
        im[:, 30*i  : 30*i+1] = 0
        im[:, 30*(i+1)-1: 30*(i+1)] = 0
        
    isFileExist = os.path.isfile(imFile)
    if isFileExist:
        print("File already exists, cannot rewrite.")
    else:
        cv2.imwrite(imFile, im)
        print("Save image success.")

def loadImage(imFile):
    im = cv2.imread(imFile)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) - 255 #将图像前景和背景反转，使得背景为0
    #plt.imshow(im2, plt.cm.gray)
    
    return im
    
def getDigit(im, i, j):
    #i, j = 3, 1 #For testing
    imx = im[imgWidth*i:imgWidth*(i+1), imgWidth*j:imgWidth*(j+1)]
    imx = imx[1:-1, 1:-1] #Remove border
    #plt.imshow(imx, plt.cm.gray)
    
    return imx
    
def grayImage24DTensor(imx):
    #将二维数字图像转为4维Tensor，以便进行二维卷积运算
    imx_4d = np.expand_dims(np.expand_dims(imx, 0), 0)
    imx_torch = torch.tensor(imx_4d).float()
    
    return imx_torch
    
fhDigitImageFile = "C:/fh/ws/ws1/fh-script/pythonsc/moai/digitrecon/digit28_28.png"
fhDigitImage = loadImage(fhDigitImageFile)

