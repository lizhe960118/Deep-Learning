# from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # an affine operation: y = Wx + b
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

input = Variable(torch.randn(1, 1, 32, 32))

# 设置目标值
target = Variable(torch.arange(1, 11))
target = target.view(1, -1)

# 设置损失函数
criterion = nn.MSELoss()

# 设置权重更新策略
optimizer = optim.SGD(net.parameters(), lr=0.001 * 9)
optimizer.zero_grad()

for i in range(10):
    # 前向传播
    output = net(input)

    # 计算损失
    loss = criterion(output, target)
    print('step %d, the loss is %0.5f' % (i + 1, loss.item()))

    # 反向传播
    loss.backward()

    # 更新权重
    optimizer.step()

    # 梯队置零，防止下一次叠加
    optimizer.zero_grad()

