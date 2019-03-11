import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F 
from torch.optim import Adam
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from data_utils import load_data
import os
import shutil
import numpy as np
import math

class MaxPoolPad(nn.Module):
    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.zeropad = nn.ZeroPad2d((1, 0, 1, 0))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.zeropad(x)
        out = self.maxpool(out)
        return out

class AvgPoolPad(nn.Module):
    def __init__(self, stride, padding):
        super(AvgPoolPad, self).__init__()
        self.zeropad = nn.ZeroPad2d((1, 0, 1, 0))
        self.avgpool = nn.AvgPool2d(3, stride=stride, padding=padding, count_include_pad=False)

    def forward(self, x):
        out = self.zeropad(x)
        out = self.avgpool(x)
        return out 

# 普通卷积使用 out_channel 个 3 * 3 * in_channel 卷积提取特征
# 参数大小 out_channel * 3 * 3 * in_channel 

# 可分离卷积先使用 in_channel 个3 * 3 * 1分别卷积每一个channel, 
# 得到和原来大小相同但是经过特征提取的特征图
# 再使用 out_channel 个 1 * 1 * in_channel卷积进行融合
# 参数大小 in_channel * 3 * 3 + out_channel * in_channel
class SeparableConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, dw_kernel_size, dw_stride, dw_padding):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channel, in_channel, dw_kernel_size, stride=dw_stride, padding=dw_padding, groups=in_channel)
        # groups = in_channel 实现 通道分离
        self.pointwiseConv2d = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.depthwise_conv2d(x)
        out = self.pointwiseConv2d(out)
        return out 
