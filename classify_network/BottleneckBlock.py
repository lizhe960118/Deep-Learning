import torch
import torch.nn as nn

import math

# 卷积层的参数数量取决于卷积核大小k，输入通道数量x， 输出通道数y
# 瓶颈块通过一个确定的比例r采用代价小的1*1卷积来减少通道数，以便后面的3*3卷积具有较少的通道数，
# 最后使用1*1卷积还原通道数

# 瓶颈块
class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        # 实际的out_channel为 这里输入的out_channel 的4倍
        # 当 in_channel = out_channel * 4 时，立刻可以直接相加
        super(BottleneckBlock, self).__init__()
        # 1 * 1 卷积 缩小想输出的维度
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel))
        
        self.relu = nn.ReLU(inplace=True)

        # 特征图可能大小的改变发生在这里
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel))

        # 1 * 1 、卷积还原通道数
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1),
            nn.BatchNorm2d(out_channel * self.expansion)
            )

        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        residual = x 

        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out 

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        # 输入 224 * 224 * 3
        super(ResNet,self).__init__()

        self.in_channel = 64

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            # (224 - 7 + 3 * 2) // 2 + 1 = 112.5
            # 舍弃 ？
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # (112 - 3 + 2 * 1)// 2 + 1 = 57?
            # 56 * 56 * 64
            )

        # 传入瓶颈块走一下流程
        # 传入需要重复的基础块， 传入需要输出的通道数， 传入基础块需要循环的次数
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 第一次 x = 56 * 56 * 64, layers[0] = 3
        # f(x) => (56 - 3 + 2 * 1)/ 1 + 1 = 56 (卷积两次形状不变)， out_channel = 64 * 4
        # 输出 56 * 56 * 64 * 4

        # 也就是说，特征图的长宽大小由stride决定，通道数由out_channel决定
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        # 输出 (56 - 3 + 2 * 1)// 2 + 1= 28 out_channel = 128 * 4
        # 输出 28 * 28 * 128 * 4
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        # 输出 14 * 14 * 256 * 4
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
        # 输出 7 * 7 * 512 * 4

        self.avgpool = nn.AvgPool2d(7)
        # 输出512
        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion, num_classes),
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

    def _make_layer(self, block, out_channel, blocks, stride=1):
        # channel 需要输出的通道数，blocks 需要叠加几次block
        downsample = None
        if stride != 1 or self.in_channel != out_channel * block.expansion:
            # 第一次：stride = 1， self.in_channel = 64, out_channel * block.expansion = 64 * 4
            # 第二次: stride = 2, self.in_channel = 64 * 4, out_channel * block.expansion = 128 * 4
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel * block.expansion, 
                    kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel * block.expansion)
                )
        layers = []

        layers.append(block(self.in_channel, out_channel, stride, downsample))
        
        # BottleneckBlock这里展宽为4
        self.in_channel = out_channel * block.expansion
        # 第一次 64 * 4
        # 第二次 128 * 4

        for i in range(1, blocks):
            layers.append(block(self.in_channel, out_channel))
            # 第一次 block(64 * 4, 64)
            # 第二次 block(128 * 4, 128)

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(out.size[0], -1)
        out = self.fc(out)

        return out 


# def resnet18(num_classes=1000):
#     model = ResNet(ResNetBasicBlock, layers=[2, 2, 2,2], num_classes)
#     return model 

def resnet50(num_classes=1000):
    model = ResNet(BottleneckBlock, layers=[3, 4, 6, 3], num_classes=num_classes)
    return model 

resnet50 = resnet50()