#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/1 21:49
@Author  : LI Zhe
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from logger import Logger

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义网络参数
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset  = torchvision.datasets.MNIST(root='./../data/MNIST',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
'''
test_dataset  = torchvision.datasets.MNIST(root='./../data/MNIST',
                                            train=False,
                                            transform=transforms.ToTensor())

'''
# 定义数据集加载函数 Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

'''
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
'''

# 包含一个隐藏层的全连接层
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

logger = Logger('C://TensorBoard/logs')
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

data_iter = iter(train_loader)
iter_per_epoch = len(train_loader)
total_step = num_epochs * iter_per_epoch

# Train the model
for step in range(total_step):

    # 每一轮训练重新加载数据
    if (step + 1) % iter_per_epoch == 0:
        data_iter = iter(train_loader)

    #  reshape images to (batch, input_size)
    images, labels = next(data_iter)
    images = images.view(images.size(0), -1).to(device)
    labels = labels.to(device)

    # forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # compute accuracy
    _, _argmax = torch.max(outputs, 1)
    accuracy = (labels == _argmax.squeeze()).float().mean()

    if (step + 1) % 100 == 0:
        print('Step:[{}/{}],loss:{:.4f}, Acc:{:.2f}'.format(step+1,total_step,loss.item(), accuracy.item()))

        # TensorBoard Logging
        # 1. log scalar values(scalar summary)
        info = {'loss':loss.item(), 'accuracy':accuracy.item()}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step+1)
        # 2. log values and gradients of the parameters (histogram summary)
        for tag, value in model.named_parameters():
            logger.histo_summary(tag, value.data.cpu().numpy(), step+1)
            logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step+1)
        # 3. log training images (images summary)
        info = {'images': images.view(-1, 28, 28)[:10].cpu().numpy()}
        for tag, images in info.items():
            logger.image_summary(tag, images, step+1)
        # 4. use tensorboard
        # tensorboard --logdir=C:\TensorBoard\logs --host=127.0.0.1
        # 路径不能出现中文 错误使用：tensorboard --logdir=C:\user\李哲\TensorBoard\logs
        # 没有引号 错误使用：tensorboard --logdir='C:\TensorBoard\logs' --host=127.0.0.1
        # 注意window下本地路径的输入 可以使用： tensorboard --logdir=C://TensorBoard/logs --host=127.0.0.1

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')