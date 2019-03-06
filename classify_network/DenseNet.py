import torch
import torch.nn as nn
# import torch.nn.functional as F 
import math

# dense块是一个residual的极端版本，其中每个卷积层都会和这个块之前所有卷积块的输出相连。
# https://blog.csdn.net/u013841196/article/details/80725764
class BottleneckBlock(nn.Module):
    expansion = 4

    # 这里只进行同大小的卷积，不进行缩放
    def __init__(self, in_channel, growthRate):
        # out_channel = growthRate
        # growthRate表示每个denseBlock中每层输出的featureMap个数
        super(BottleneckBlock, self).__init__()
        # 1 * 1 卷积缩小输出通道 
        # 例如第32层（31 * growthRate) =>(growthRate * 4)
        # BN-ReLU-Conv
        self.conv_layer1 = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channel, growthRate * 4, kernel_size=1),
            )

        # 这里每个dense block的3*3卷积前面都包含了一个1*1的卷积操作，就是所谓的bottleneck layer
        # 3 * 3 卷积
        self.conv_layer2 = nn.Sequential(
            nn.BatchNorm2d(growthRate * 4),
            nn.ReLU(inplace = True),
            nn.Conv2d(growthRate * 4, growthRate, kernel_size=3, padding=1),
            )

        # # 1 * 1 卷积恢复
        # self.conv_layer3 = nn.Sequential(
        #     nn.Conv2d(out_channel // self.expansion, out_channel, kernel_size=1),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU(inplace = True)
        #     )

    def forward(self, x):
        residual = x
        out = self.conv_layer1(x)
        out = self.conv_layer2(x)
        # 在channel上将他们拼接起来
        out = torch.cat((residual, out), 1)
        return out 

class BasicBlock(nn.Module):
    # 未使用1 * 1卷积降维， 只经过一次 3*3 卷积之后， 原来的输入图片拼接
    def __init__(self, in_channel, growthRate):
        super(SingleLayer, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channel,growthRate, kernel_size=3, padding=1),
            )

    def forward(self, x):
        residual = x
        out = self.conv_layer(x)
        out = torch.cat((residual, out), 1)
        return out 

class TransitionLayer(nn.Module):
    # 将之前所有concat的层做一次1*1卷积降维

    # 虽然第32层的3*3卷积输出channel只有32个（growth rate），但是紧接着还会像前面几层一样有通道的concat操作，
    # 即将第32层的输出和第32层的输入做concat，前面说过第32层的输入是1000左右的channel，
    # 所以最后每个Dense Block的输出也是1000多的channel。
    # 因此这个transition layer有个参数reduction（范围是0到1），表示将这些输出缩小到原来的多少倍，默认是0.5，
    # 这样传给下一个Dense Block的时候channel数量就会减少一半，这就是transition layer的作用。

    # 使之输出通道数为out_channel = in_channel * reduction
    # 最后做一次平均池化,输出通道数
    def __init__(self, in_channel, out_channel):
        super(TransitionLayer, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.AvgPool2d(kernel_size=2,stride=2)
            )
    def forward(self, x):
        out = self.conv_layer(x)
        return out
    

class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, num_classes, useBottleneck):
        super(DenseNet, self).__init__()
        if useBottleneck:
            # 计算 dense_block的数量 
            # （卷积conv1，pooling1）, TransitionLayer(3), （pool_last, fc_last）
            # depth = 5 + 3 * 2 * n_blocks = 65
            n_blocks = (depth - 5) // 2
            
        else:
            n_blocks = (depth - 5) // 1

        # 3个denseBlocks, 计算每个denseBlock中的bottleneck/basicBlock的个数
        n_blocks = n_blocks // 3
        # 假设每一个denseblock有10个bottleneck

        # 输入 224 * 224 * 3
        in_channel = 2 * growthRate

        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, in_channel, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        # 56 * 56 * 64

        self.dense1 = self._make_dense(in_channel, growthRate, n_blocks, useBottleneck)
        in_channel += growthRate * n_blocks
        # 56 * 56 * (64 + 32 * 10) = 56 * 56 * 384

        out_channel = int(in_channel * reduction)
        self.trans1 = TransitionLayer(in_channel, out_channel)
        # 56 * 56 * 384 = > 28 * 28 * 192
        in_channel = out_channel

        self.dense2 = self._make_dense(in_channel, growthRate, n_blocks, useBottleneck)
        in_channel += growthRate * n_blocks
        # 28 * 28 * 192 => 28 * 28 * (192 + 320) => 28 * 28 * 512

        out_channel = int(in_channel * reduction)
        self.trans2 = TransitionLayer(in_channel, out_channel)
        # 28 * 28 * 512 => 14 * 14 * 256
        in_channel = out_channel

        self.dense3 = self._make_dense(in_channel, growthRate, n_blocks, useBottleneck)
        in_channel += growthRate * n_blocks
        # 14 * 14 * 256 => 14 * 14 * (256 + 320) = 14 * 14 * 576

        # out_channel = in_channel * reduction
        # self.trans3 = self.TransitionLayer(in_channel, out_channel)
        # # 14 * 14 * 576 => 7 * 7 * 288
        # in_channel = out_channel

        self.avg_pool = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.AvgPool2d(7)
            )
        
        self.fc = nn.Sequential(
            nn.Linear(in_channel, num_classes),
            nn.ReLU(),
            nn.Softmax()
            )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, in_channel, growthRate, n_blocks, useBottleneck):
        layers = []

        for i in range(int(n_blocks)):
            if useBottleneck:
                layers.append(BottleneckBlock(in_channel, growthRate))
            else:
                layers.append(BasicBlock(in_channel, growthRate))
            in_channel += growthRate

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.avg_pool(self.dense3(out))
        out = torch.squeeze(out)
        out = self.fc(out)
        return out 

def densenet65_32(num_classes=10):
    model = DenseNet(32, 65, 0.5, num_classes=num_classes, useBottleneck=True)
    return model

densenet65_32 = densenet65_32()

# DenseNet核心思想在于建立了不同层之间的连接关系，充分利用了feature，进一步减轻了梯度消失问题，加深网络不是问题，而且训练效果非常好。
# 另外，利用bottleneck layer，Translation layer以及较小的growth rate使得网络变窄，参数减少，有效抑制了过拟合，同时计算量也减少了。

"""
def dense_block(self, x, f=32, d=5):
    l = x
    for i in range(d):
        x = conv(l, f)
        l = concatnate([l, x])
    return l
"""












