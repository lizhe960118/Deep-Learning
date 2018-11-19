#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/13 15:27
@Author  : LI Zhe
"""
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import os

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')

def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out

batch_size = 128
num_epochs = 100
learning_rate = 1e-3
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# import MNIST
mnist = datasets.MNIST(root='./../data/MNIST', train=True, transform=image_transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

# autoencoder
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = autoencoder()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        num_img = img.size(0)
        img = img.view(num_img, -1)
        img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()
        # forward pass
        out = model(img)
        loss = criterion(out, img)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch:[{}/{}], loss:{:.6f}'.format(
        epoch + 1, num_epochs, loss.data))
    if epoch % 10 == 0:
        pic = to_img(out.cpu().data)
        save_image(pic, './mlp_img/images_{}.png'.format(epoch))

torch.save(model.state_dict(), './simple_autoencoder.pth')









