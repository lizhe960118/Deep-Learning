import torch
import torch.nn as nn
import torch.nn.functional as F

class inception_module(nn.Module):
    def __init__(self, in_channel, out_channels):
        super(inception_module, self).__init__()

        """
        # 1x1 convolution
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channels[0], kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU()
            )

        # bottleneck layer and 3x3 convolution
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channels[1], kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels[1]),
            nn.ReLU(),
            nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[2]),
            nn.ReLU()
            )

        # bottleneck layer and 5x5 convolution
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(in_channel, out_channels[3], kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(),
            nn.Conv2d(out_channels[3], out_channels[4], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channels[4]),
            nn.ReLU()
            )
        # maxpooling and bottleneck layer

        self.maxpool_3x3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channels[5], kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels[5]),
            nn.ReLU()
            )
        """
        # 1x1 convolution
        self.conv_1x1 = nn.Conv2d(in_channel, out_channels[0], kernel_size=1, stride=1)

        # bottleneck layer and 3x3 convolution
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channels[1], kernel_size=1, stride=1),
            nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, stride=1, padding=1)
            )

        # bottleneck layer and 5x5 convolution
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(in_channel, out_channels[3], kernel_size=1, stride=1),
            nn.Conv2d(out_channels[3], out_channels[4], kernel_size=5, stride=1, padding=2)
            )

        # maxpooling and bottleneck layer
        self.maxpool_3x3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel, out_channels[5], kernel_size=1, stride=1)
            )

    def forward(self, x):
        conv_1x1_out = self.conv_1x1(x)
        conv_3x3_out = self.conv_3x3(x)
        conv_5x5_out = self.conv_5x5(x)
        maxpool_3x3_out = self.maxpool_3x3(x)
        out = torch.concat([conv_1x1_out, conv_3x3_out, conv_5x5_out, maxpool_3x3_out], 1)
        return out 

class aux_classifier(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(aux_classifier, self).__init__()
        # Average pooling, fc, dropout, fc
        self.average_pool = nn.AvePool2d(kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv2d(in_channel, 128, kernel_size=1, stride=1)
        # self.fc_1 = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(p=0.7)
        self.fc_2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.average_pool(x)
        out = self.conv(out)
        out = out.reshape(out.size(0), -1)

        self.fc_1 = nn.Linear(out.size(-1),1024)

        out = self.fc_1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc_2(out)
        out = F.softmax(out)
        return out 

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()

        # stage 1 - layers before inception modules
        # 输入 32 * 32 * 3
        self.stage_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 16 * 16 * 64
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            # 16 * 16 * 192
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # 16 * 16 * 192
            )

        # stage2 - 2 inception modules and max pooling
        self.stage_2_1 = inception_module(192, out_channels=[64, 96, 128, 16, 32, 32])
        # 16 * 16 * 256
        self.stage_2_2 = inception_module(256, out_channels=[128, 128, 192, 32, 96, 64])
        # 16 * 16 *  480
        self.stage_2_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 8 * 8 *  480

        # stage3 - 5 inception modules and max pooling
        self.stage_3_1 = inception_module(480, out_channels = [192, 96, 208, 16, 48, 64])
        # 8 * 8 * 512
        self.stage_3_aux_1 = aux_classifier(512, num_classes)

        self.stage_3_2 = inception_module(512, out_channels = [160, 112, 225, 24, 64, 64])
        self.stage_3_3 = inception_module(512, out_channels = [128, 128, 256, 24, 64, 64])
        # 8 * 8 * 512
        self.stage_3_4 = inception_module(528, out_channels = [112, 144, 288, 32, 64, 64])
        # 8 * 8 * 528
        self.stage_3_aux_2 = aux_classifier(528, num_classes)

        self.stage_3_5 = inception_module(528, out_channels = [256, 160, 320, 32, 128, 128])
        # 8 * 8 * 832
        self.stage_3_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 4 * 4 * 832

        # stage4 - 2 inception modules and average pooling
        self.stage_4_1 = inception_module(832, out_channels = [256, 160, 320, 32, 128, 128])
        # 4 * 4 * 832
        self.stage_4_2 = inception_module(832, out_channels = [384, 192, 384, 48, 128, 128])
        # 4 * 4 * 1024
        self.stage_4_avgpool = nn.AvgPool(kernel_size=4, stride=1)
        # 1 * 1 * 1024

        # stage5 - dropout,linear fc, softmax fc
        self.stage_5_drop = nn.Dropout(0.4)
        self.stage_5_fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.stage_1(x)

        out = self.stage_2_1(out)
        out = self.stage_2_2(out)
        out = self.stage_2_maxpool(out)

        out = self.stage_3_1(out)
        auc_output_1 = self.stage_3_aux_1(out)
        out = self.stage_3_2(out)
        out = self.stage_3_3(out)
        out = self.stage_3_4(out)
        auc_output_2 = self.stage_3_aux_2(out)
        out = self.stage_3_5(out)
        out = self.stage_3_maxpool(out)

        out = self.stage_4_1(out)
        out = self.stage_4_2(out)
        out = self.stage_4_avgpool(out)

        out = out.reshape(out.size(0), -1)
        out = self.stage_5_drop(out)
        out = self.stage_5_fc(out)
        out = F.softmax(out)

        return [out, auc_output_1, auc_output_2]

device = ""
learning_rate = 0.001
model = GoogLeNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, beta=[0.9,0.99], eps=1e-8, decay=0)

