import torch 
import torch.nn as nn

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResidualBlock, self).__init__()

        # 先通过1*1的卷积块压缩
        self.conv_1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // self.expansion, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // self.expansion),
            nn.ReLU()
            )
        # 普通 3 * 3 卷积
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels // self.expansion, in_channels // self.expansion, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // self.expansion),
            nn.ReLU()
            )
        self.conv_1x1_2 = nn.Sequential(
            nn.Conv2d(in_channels // self.expansion, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )

        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv_1x1_1(x)
        out = self.conv_3x3(out)
        out = self.conv_1x1_2(out)

        if self.downsample:
            residual = self.downsample(residual)

        out = out + residual
        return out

class InvertedResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, downsample=None):
        super(InvertedResidualBlock, self).__init__()

        # 先通过1*1的卷积块扩张
        self.conv_1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * self.expansion, kernel_size=1),
            nn.BatchNorm2d(in_channels * self.expansion),
            nn.ReLU()
            )
        # 通过3*3的可分离卷积
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels * self.expansion, in_channels * self.expansion, kernel_size=3, stride=1, padding=1, groups=in_channels * self.expansion),
            nn.BatchNorm2d(in_channels * self.expansion),
            nn.ReLU()
            )
        self.conv_1x1_2 = nn.Sequential(
            nn.Conv2d(in_channels * self.expansion, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )

        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv_1x1_1(x)
        out = self.conv_3x3(out)
        out = self.conv_1x1_2(out)

        if self.downsample:
            residual = self.downsample(residual)

        out = out + residual
        return out