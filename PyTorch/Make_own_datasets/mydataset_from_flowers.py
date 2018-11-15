#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/15 14:01
@Author  : LI Zhe
"""
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt

img_data = torchvision.datasets.ImageFolder('./../data/flower_photos',
                                            transform=transforms.Compose([
                                                transforms.Scale(256),
                                                transforms.CenterCrop(224), # 中心化
                                                transforms.ToTensor()
                                            ])
                                            )
print(len(img_data))
data_loader = DataLoader(img_data, batch_size=20, shuffle=True)
print(len(data_loader))

def show_batch(imgs):
    grid = utils.make_grid(imgs, nrow=5)
    plt.imshow(grid.numpy().transpose(1, 2, 0))
    plt.title('Batch from dataloader')

for i, (batch_x, batch_y) in enumerate(data_loader):
    if i < 4:
        print(i, batch_x.size(), batch_y.size())
        show_batch(batch_x)
        print(batch_y)
        plt.axis('off')
        plt.show()