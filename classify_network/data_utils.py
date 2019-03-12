import torch
import torchvision
from torchvision import transforms
# import matplotlib.pyplot as plt
import os
# import numpy as np

def load_data(data_folder, net_name):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    if net_name == "alexnet":
        train_transform = transforms.Compose(
            [transforms.RandomSizedCrop(227),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
            [transforms.Scale(256),transforms.CenterCrop(227),transforms.ToTensor(),transforms.Normalize(mean, std)])
    elif net_name == "vgg16net" or net_name =="resnet" or net_name=="resnet_bottleneck" or net_name == "densenet" or net_name == "senet":
        train_transform = transforms.Compose(
            [transforms.RandomSizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
            [transforms.Scale(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean, std)])        
    else: #net_name =="resnext" or net_name=="googlenet"
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])
        

    train_data = torchvision.datasets.CIFAR10(data_folder, train=True, transform=train_transform, download=True)
    test_data = torchvision.datasets.CIFAR10(data_folder, train=False, transform=test_transform, download=True)
    # num_classes = 10 

    return train_data, test_data

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

