#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/15 13:23
@Author  : LI Zhe
"""

import torch
import torchvision
import matplotlib.pyplot as plt
from skimage import io
import os

mnist_test = torchvision.datasets.MNIST('./../data/MNIST', train=False, download=True)
print('test set:', len(mnist_test))

if not os.path.exists('./mnist_test'):
    os.mkdir('./mnist_test')

f = open('mnist_test.txt', 'w')
for i, (img, label) in enumerate(mnist_test):
    img_path = './mnist_test/' + str(i) + '.jpg'
    # io.imsave(img_path, img)
    f.write(img_path + ' ' + str(label.item()) + '\n')
f.close()