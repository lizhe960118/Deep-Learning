import torch
import torch.nn as nn

import math

# ResNet作者指出，增加网络深度会导致更高的训练误差，
# 这表明梯度问题（梯度消失/爆炸）可能会导致训练收敛性等潜在问题。
# ResNet 的主要贡献是增加了神经网络架构的跳过连接（skip connection），使用批归一化并移除了作为最后一层的全连接层。

# 除了跳过链接，每次卷积完成之后，激活进行之前都采取了批归一化
# 最后，网络删除了全连接层，并使用平均池化层减少参数的数量。
# 网络加深，卷积层的抽象能力变强，从而减少了对全连接层的需求。

# 基础残差块
class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(ResNetBasicBlock, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel))
        
        self.relu = nn.ReLU(inplace=True)

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel))

        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        residual = x 

        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out 

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        # 输入 224 * 224 * 3
        super(ResNet,self).__init__()
        # 首先找到ResNet的父类（比如是类nn.Module），然后把类ResNet的对象self转换为类nn.Module的对象，
        # 然后“被转换”的类nn.Module对象调用自己的__init__函数
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

        # 传入需要重复的基础块， 传入需要输出的通道数， 传入基础块需要循环的次数
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 第一次 x = 56 * 56 * 64
        # f(x) => (56 - 3 + 2 * 1)/ 1 + 1 = 56 (卷积两次形状不变)， out_channel = 64
        # 输出 56 * 56 * 64
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        # 输出 (56 - 3 + 2 * 1)// 2 + 1= 28
        # 输出 28 * 28 * 128
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        # 输出 14 * 14 * 256
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
        # 输出 7 * 7 * 512

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
        # out_channel 需要输出的通道数，blocks 需要叠加几次block
        downsample = None
        if stride != 1 or self.in_channel != out_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel * block.expansion, 
                    kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel * block.expansion)
                )
        layers = []

        layers.append(block(self.in_channel, out_channel, stride, downsample))
        
        # BasicBlock这里展宽为1
        self.in_channel = out_channel * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.in_channel, out_channel))

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


def resnet18(num_classes=1000):
    model = ResNet(ResNetBasicBlock, layers=[2,2, 2,2], num_classes=num_classes)
    return model 

def resnet50(num_classes=1000):
    model = ResNet(BottleneckBlock, layers=[3, 4, 6, 3], num_classes=num_classes)
    return model 


resnet18 = resnet18()