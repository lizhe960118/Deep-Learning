#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/13 16:56
@Author  : LI Zhe
"""
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import os

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out

batch_size = 128
num_epochs = 100
z_dimension = 100
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# import MNIST
mnist = datasets.MNIST(root='./../data/MNIST', train=True, transform=image_transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True, num_workers=4)

# Discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), # batch, 32, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2) # batch, 32, 14, 14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2), # batch, 64, 14, 14
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2) # batch, 64, 7, 7
        )
        self.fc= nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Generator
class generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)  # batch, 3136=1*56*56
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),  # batch, 50, 56, 56
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),  # batch, 25, 56, 56
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, stride=2),  # batch, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x

D = discriminator()
G = generator(z_dimension, 3136)

if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

for epoch in range(num_epochs):
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        # -------- train discriminator---------
        # img = img.view(num_img, -1)
        real_img = Variable(img).cuda()
        real_label = Variable(torch.ones(num_img)).cuda()
        fake_label = Variable(torch.zeros(num_img)).cuda()

        # compute loss of real_img
        real_out = D(real_img)
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out

        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out

        # backward and optimize
        d_loss = d_loss_fake + d_loss_real
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # -------- train generator ---------
        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        output = D(fake_img)
        g_loss = criterion(output, real_label)

        # backward and optimize
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch:[{}/{}], d_loss:{:.6f},g_loss:{:.6f},D_real:{:.6f},D_fake:{:.6f}'.format(
                epoch, num_epochs, d_loss.data[0], g_loss.data[0], real_scores.data.mean(), fake_scores.mean()))

    if epoch == 0:
        real_images = to_img(real_img.cpu().data)
        save_image(real_images, './dc_img/real_images.png')
    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, './dc_img/fake_inages-{}.png'.format(epoch + 1))

torch.save(G.state_dict(), './dc_img/generator.pth')
torch.save(D.state_dict(), './dc_img/discriminator.pth')
