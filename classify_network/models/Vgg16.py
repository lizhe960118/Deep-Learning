import torch 
import torch.nn as nn
"""
VGG 的优点在于，堆叠多个小的卷积核而不使用池化操作可以增加网络的表征深度，同时限制参数的数量。
例如，通过堆叠 3 个 3×3 卷积层而不是使用单个的 7×7 层，可以克服一些限制。
首先，这样做组合了三个非线性函数，而不只是一个，使得决策函数更有判别力和表征能力。
第二，参数量减少了 81%，而感受野保持不变。
另外，小卷积核的使用也扮演了正则化器的角色，并提高了不同卷积核的有效性。

VGG 的缺点在于，其评估的开销比浅层网络更加昂贵，内存和参数（140M）也更多。
这些参数的大部分都可以归因于第一个全连接层。
结果表明，这些层可以在不降低性能的情况下移除，同时显著减少了必要参数的数量。
"""

class VGG16_Net(nn.Module):
    def __init__(self):
        super(VGG16_Net, self).__init__()

        self.layer1 = nn.Sequential(
            # 核心思想用小卷积核代替大卷积核，减少参数
            # 卷积 1-1
            # 输入 （224 * 224 * 3） 卷积（3 * 3 * 3）步长 1,填充1， 个数 64
            # 输出 （224 - 3 + 2 * 1）/ 1 + 1 = 224 （224 * 224 * 64）
            # 归一化BN
            # relu处理 （relu提高训练速度）
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 卷积 1-2
            # 输入 （224 * 224 * 64） 卷积（3 * 3 * 3）步长 1,填充1， 个数 64
            # 输出 （224 - 3 + 2 * 1）/ 1 + 1 = 224 （224 * 224 * 64）
            # 归一化BN
            # relu处理
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # maxpool - 1 
            # kernel_size = 2, 步长2
            # （224 * 224 * 64）=> )(112, 112, 64)
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.layer2 = nn.Sequential(
            # 112 * 112 * 64 => 112 * 112 * 128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # conv 2 -2
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # maxpool (112 * 112 * 128) => (56 * 56 *128)
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer3 = nn.Sequential(
            # (56 * 56 * 128) = (56 * 56 * 256)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # conv 3-2
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # conv 3-3
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # max_pool (56 * 56 *256 => 28 * 28 * 256)
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer4 = nn.Sequential(
            # (28 * 28 * 256) = (28 * 28 * 512)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # conv 4-2
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # conv 4 -3        
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # max_pool (28 * 28 * 512 => 14 * 14 * 512)
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer5 = nn.Sequential(
            # (14 * 14 * 512) => (14 * 14 * 512)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # conv 5-2
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # conv 5-3        
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # max_pool (14 * 14 * 512 => 7 * 7 * 512)
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer6 = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU())
        self.layer8 = nn.Sequential(
            nn.Linear(4096, 1000),
            nn.BatchNorm1d(1000),
            nn.Softmax())


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        return out 

# vgg16 = VGG16_Net()

class vgg16_model(object):
    