#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/1 16:42
@Author  : LI Zhe
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 定义网络参数
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset  = torchvision.datasets.MNIST(root='./../data/MNIST',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
test_dataset  = torchvision.datasets.MNIST(root='./../data/MNIST',
                                            train=False,
                                            transform=transforms.ToTensor())

# 定义数据集加载函数 Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# Logistic regression model
model = nn.Linear(input_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #  reshape images to (batch, input_size)
        images = images.reshape(-1, 28* 28)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch:[{}/{}], Step:[{}/{}],loss:{:.4f}'.format(epoch+1,num_epochs,i+1,total_steps,loss.item()))

# Test model
with torch.no_grad():
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        #  reshape images to (batch, input_size)
        images = images.reshape(-1, 28 * 28)

        # forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        print('Accuracy of the model on the 10000 test images: {}%'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

