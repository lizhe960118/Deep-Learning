import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


"""
# 测试
# print(torch.__version__)
w = torch.ones(1, requires_grad=True)
input = torch.rand(1, 10, requires_grad=True)
print(w.requires_grad)
print(input.requires_grad)
"""

"""
自动求导：例1
"""
'''
# 构造张量
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# 创建一个计算图
y = w * x + b  # y = 2 * x + 3

# 计算梯度
y.backward()

# 显示梯度
print(x.grad) # 2
print(w.grad) # 1
print(b.grad) # 1
'''

"""
自动求导：例2
"""
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# 搭建一个全连接网络
linear = nn.Linear(3, 2)
print('w:', linear.weight)
print('b:', linear.bias)

# 设置损失函数和梯度优化策略
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(),lr=0.01)

# 前向传播
pred = linear(x)

# 计算损失
loss = criterion(pred, y)
print('loss:', loss.item())

# 反向传播
loss.backward()

# 输出梯度
print('w_grad:', linear.weight.grad)
print('b_grad:', linear.bias.grad)

# 梯度更新
optimizer.step()

# 再次前向传播，计算损失
pred = linear(x)
loss = criterion(pred, y)
print('loss after one step optimization:', loss.item())

