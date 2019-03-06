#ResNeXt将ResNet中的操作重复基数多次，从而增加参数的利用率

# 原来是 28 * 28 * 256
# 引入基数32，压缩率为2， 使用32个平行的卷积层，使其输出通道为4
# 之后32个 28 * 28 * 4 的特征图进行3 * 3卷积
# 最后 28 * 28 * 4的特征图进行 1*1卷积，输出 28 * 28 * 256
# 将32个特征图相加，加上残差，得到输出
import torch
import torch.nn as nn
import math

class ResNeXtBottleBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, cardinality=16, stride=1, downsample=None):
        super(ResNeXtBottleBlock, self).__init__()

        block_channel = in_channel // (cardinality * self.expansion)
        
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channel, block_channel * cardinality, kernel_size=1),
            nn.BatchNorm2d(block_channel * cardinality),
            nn.ReLU()
            )
        self.conv_conv = nn.Sequential(
            nn.Conv2d(block_channel * cardinality, block_channel * cardinality, kernel_size=3, stride=stride, groups=cardinality),
            # groups: 控制输入和输出之间的连接，
            # group=1，输出是所有的输入的卷积；
            # group=2，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。
            nn.BatchNorm2d(block_channel * cardinality),
            nn.ReLU()
            )
        self.conv_expand = nn.Sequential(
            nn.Conv2d(block_channel * cardinality, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
            )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv_reduce(x)
        out = self.conv_conv(out)
        out = self.conv_expand(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        return out 

class ResNeXt(nn.Module):
    def __init__(self, block, depth, cardinality, num_classes):
        super(ResNeXt,self).__init__()

        n_blocks = int((depth - 2) // 9)
        # 计算残差块的数量
        self.cardinality = cardinality

        # 输入 32 * 32 * 3
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            # 32 * 32 * 3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            )

        self.in_channel = 64
        # 传入需要重复的基础块， 传入需要输出的通道数， 传入基础块需要循环的次数
        self.layer1 = self._make_layer(block, 64, n_blocks)
        # 32 * 32 * 64 => 32 * 32 * 64

        self.layer2 = self._make_layer(block, 128, n_blocks, stride = 2)
        # 输出 16 * 16 * 128
        self.layer3 = self._make_layer(block, 256, n_blocks, stride = 2)
        # 输出 8 * 8 * 256

        self.avgpool = nn.AvgPool2d(7)
        # 输出256

        self.fc = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.ReLU(),
            nn.Softmax()
            ) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_channel, n_blocks, stride=1):
        # out_channel 需要输出的通道数，blocks 需要叠加几次block
        downsample = None
        if stride != 1 or self.in_channel != out_channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel, 
                    kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
                )

        layers = []

        layers.append(block(self.in_channel, out_channel, self.cardinality, stride, downsample))
        
        self.in_channel = out_channel

        for i in range(1, n_blocks):
            layers.append(block(self.in_channel, out_channel, self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.reshape(out.size[0], -1)
        out = self.fc(out)

        return out 


def resnext29_16(num_classes=10):
    model = ResNeXt(ResNeXtBottleBlock, depth=29, cardinality=16, num_classes=num_classes)
    return model 

resnext29_16 = resnext29_16()
