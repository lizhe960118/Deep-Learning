#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/16 22:55
@Author  : LI Zhe
"""

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F
import os

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')

def to_img(x):
    out = x.clamp(0, 1)
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
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar

model = autoencoder()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

def loss_function(recon_x, x, mu, logvar):
    BCE = criterion(recon_x, x)
    KLD_elenment = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_elenment).mul_(-0.5)
    return BCE + KLD

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        img, _ = data
        num_img = img.size(0)
        img = img.view(num_img, -1)
        img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()
        # forward pass
        recon_out, mu, logvar = model(img)
        loss = loss_function(recon_out, img, mu, logvar)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data

        if batch_idx % 100 == 0:
            print('Epoch:{} [{}/{} ({:.0f}%)]\tloss:{:.6f}'.format(
                epoch + 1, batch_idx * len(img),
                len(dataloader.dataset),
                100.0 * batch_idx / len(dataloader),
                loss.data[0] / len(img)))

    print('Epoch {} Finished, Average loss:{:.4f}'.format(epoch + 1, train_loss / len(dataloader.dataset)))

    if epoch % 10 == 0:
        pic = to_img(recon_out.cpu().data)
        save_image(pic, './vae_img/images_{}.png'.format(epoch))

torch.save(model.state_dict(), './variational_autoencoder.pth')