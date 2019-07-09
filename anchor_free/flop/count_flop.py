#!/usr/bin/env python
# coding=utf-8
#import torchvision.models as models
import torch
from thop import profile

#print(torch.__version__)
#model = models.inception_v3()

from retinanet import RetinaNet
model = RetinaNet()
#input = torch.randn(1, 3, 224, 224)
input = torch.randn(1, 3, 600, 600)
flops, params = profile(model, inputs = (input,))

print("params:", params)
print("flops:", flops)

