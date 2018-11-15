#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/13 21:41
@Author  : LI Zhe
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

# 定义网络参数
num_epochs = 20
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

# 定义循环神经网络
class Recurrent_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layer, num_classes):
        super(Recurrent_Net, self).__init__()
        self.n_layer1 = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self,x):
        out, _  = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out

model = Recurrent_Net(28, 128, 2, 10)

# device_ids = [0,1]
if torch.cuda.device_count() > 1:
    print("let's use",torch.cuda.device_count(), 'GPUs!')
    model = nn.DataParallel(model)
#     model = nn.DataParallel(model, device_ids=device_ids)

if torch.cuda.is_available():
    model = model.cuda()

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

# Train the model
for epoch in range(num_epochs):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        image, label = data
        b, c, h, w = image.size()
        assert (c == 1, 'channel must be 1')
        image = image.squeeze(1)

        if torch.cuda.is_available():
            image = Variable(image).cuda()
            label = Variable(label).cuda()
        else:
            image = Variable(image)
            label = Variable(label)

        # forward pass
        output = model(image)
        loss = criterion(output, label)
        running_loss += loss.data * label.size(0)
        _, pred = torch.max(output, 1)
        num_correct = (pred == label).sum().item()
        running_acc += num_correct

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#         optimizer.module.step()

        if (i + 1) % 300 == 0:
            print('Epoch:[{}/{}], loss:{:.6f}, acc:{:.6f}'.format(epoch+1,num_epochs,
                                                                  running_loss / (batch_size * i),
                                                                  running_acc / (batch_size * i)))
    print('Finish {} epoch, loss:{:.6f}, acc:{:.6f}'.format(epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))))

# Test model
model.eval()
# eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
eval_loss = 0.0
eval_acc = 0.0

with torch.no_grad():
    correct = 0
    total = 0
    for i, data in enumerate(test_loader, 1):
        image, label = data
        b, c, h, w = image.size()
        assert (c == 1, 'channel must be 1')
        image = image.squeeze(1)

        if torch.cuda.is_available():
            image = Variable(image).cuda()
            label = Variable(label).cuda()
        else:
            image = Variable(image)
            label = Variable(label)

        # forward pass
        output = model(image)
        test_loss = criterion(output, label)
        eval_loss += test_loss.data * label.size(0)
        _, pred = torch.max(output.data, 1)
        num_correct = (pred == label).sum().item()
        eval_acc += num_correct

        total += label.size(0)
        correct += (pred == label).sum().item()
        print('Accuracy of the model on the 100 test images: {}%'.format(100 * correct / total))
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_dataset)), eval_acc / (len(test_dataset))))

# 保存模型
# torch.save(model.state_dict(), './RNN.pth')