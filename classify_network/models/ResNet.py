import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F 
from torch.optim import Adam
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from data_utils import load_data
import os
import shutil
import numpy as np
import math


# ResNet作者指出，增加网络深度会导致更高的训练误差，
# 这表明梯度问题（梯度消失/爆炸）可能会导致训练收敛性等潜在问题。
# ResNet 的主要贡献是增加了神经网络架构的跳过连接（skip connection），使用批归一化并移除了作为最后一层的全连接层。

# 除了跳过链接，每次卷积完成之后，激活进行之前都采取了批归一化
# 最后，网络删除了全连接层，并使用平均池化层减少参数的数量。
# 网络加深，卷积层的抽象能力变强，从而减少了对全连接层的需求。

# 基础残差块
class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(ResNetBasicBlock, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel))
        
        self.relu = nn.ReLU(inplace=True)

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel))

        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        residual = x 

        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out 

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        # 输入 32 * 32 * 3
        super(ResNet,self).__init__()
        # 首先找到ResNet的父类（比如是类nn.Module），然后把类ResNet的对象self转换为类nn.Module的对象，
        # 然后“被转换”的类nn.Module对象调用自己的__init__函数
        self.in_channel = 64

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            # (224 - 7 + 3 * 2) // 2 + 1 = 112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # (112 - 3 + 2 * 1)// 2 + 1 = 57?
            # 56 * 56 * 64
            )

        # 传入需要重复的基础块， 传入需要输出的通道数， 传入基础块需要循环的次数
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 第一次 x = 56 * 56 * 64
        # f(x) => (56 - 3 + 2 * 1)/ 1 + 1 = 56 (卷积两次形状不变)， out_channel = 64
        # 输出 56 * 56 * 64
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        # 输出 (56 - 3 + 2 * 1)// 2 + 1= 28
        # 输出 28 * 28 * 128
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        # 输出 14 * 14 * 256
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
        # 输出 7 * 7 * 512

        self.avgpool = nn.AvgPool2d(7)
        # 输出512
        self.fc = nn.Sequential(
            nn.Linear(512,num_classes),
            nn.BatchNorm1d(num_classes),
            nn.Softmax()
            ) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_channel, blocks, stride=1):
        # out_channel 需要输出的通道数，blocks 需要叠加几次block
        downsample = None
        if stride != 1 or self.in_channel != out_channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
                )
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample))
        self.in_channel = out_channel
        for i in range(1, blocks):
            layers.append(block(self.in_channel, out_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out 

class resnet_model(object):

    def __init__(self, dataset_path, save_model_path, save_history_path, epochs, batchsize, device, mode):
        """
        dataset_path:'./data/train_data'
        save_path:'./data_save'
        epochs:10
        bacth_size:20
        z_dim:100
        device: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        """
        self.dataset_path = dataset_path
        self.save_model_path = save_model_path
        self.save_history_path = save_history_path
#         self.epochs = epochs
        self.epochs = 10
#         self.batch_size = batch_size
        self.batch_size = 20
        self.mode = mode
        self.learning_rate = 0.0002
        self.num_classes = 10
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda:1")
        else:
            self.device = torch.device("cpu")
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.train_data, self.test_data = load_data(self.dataset_path, net_name="resnet")

        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        self.trian_batch_nums = len(self.train_loader) // self.batch_size
        self.test_batch_nums = len(self.test_loader) // self.batch_size
        
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []
        self.cur_model_name = os.path.join(self.save_model_path, 'current_resnet_net.t7')
        self.best_model_name = os.path.join(self.save_model_path, 'best_resnet_net.t7')
        self.max_loss = 0
        self.min_loss = float("inf")
        
    # def adjust_learning_rate(self, epoch, optimizer):
    #     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #     self.learning_rate *= (0.1 ** (epoch // 30))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = self.learning_rate

    def train(self):
        resnet = ResNet(ResNetBasicBlock, layers=[3, 4, 6, 3], num_classes=self.num_classes).to(self.device)
        optimizer = Adam(resnet.parameters(), betas=(.5, 0.999), lr=self.learning_rate)
        step = 0
        for epoch in range(self.epochs):
            
#             self.adjust_learing_rate(epoch, optimizer)
            
            print("Main epoch:{}".format(epoch))
            # count the accuracy when train data
            correct = 0
            total = 0

            for i, (images, labels) in enumerate(self.train_loader):
                
                # batch_size=100,所以step = 50000/100 = 500
                step += 1

                images = images.to(self.device)
                labels = labels.to(self.device)
                # compute output
                outputs = resnet(images)
                train_loss = self.criterion(outputs, labels)

                # compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # compute gradient and do Adam step
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                print("epoch:{}, step:{}, loss:{}, accuracy:{}".format(epoch, step, train_loss.item(),(100 * correct / total)))

                #show result and save model 
                # step = epochs * (50000 // batch_size) = 100 * 50000 // 100 = 50000
                # if (step % 100) == 0:
            # each epoch store the history and accurate
            self.train_loss_history.append(train_loss)
            self.max_loss = max(self.max_loss, train_loss)
            self.min_loss = min(self.min_loss, train_loss)
            # 计算模型在训练集上准确率
            train_accuracy = 100 * correct / total 
            self.train_accuracy_history.append(train_accuracy)
            
            # 保存当前模型
            torch.save(resnet, self.cur_model_name)
            
            # 计算当前模型在测试集上的准确率的准确率
            val_accuracy = self.validate()
            self.val_accuracy_history.append(val_accuracy)

            # 如果准确率高，则替换最佳模型为当前模型
            if epoch > 0 and val_accuracy > max(self.val_accuracy_history[:-1]):
                # 每一轮训练之后进行验证，如果平均准确率高
                # 这里加上准确率的判断，如果在验证集的前多少取得的准确率较高，则替换模型
                shutil.copyfile(self.cur_model_name, self.best_model_name) # 把cur_model复制给best_model
       
        # 调用print_history将loss画出来并保存为图片
        # print_accarucy将在验证集上的准确率保存下来，存到图片里
        self.plot_curve()

    def validate(self):
        resnet = torch.load(self.cur_model_name).to(self.device)
        # Test model
        resnet.eval()

        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(self.test_loader):
                #  reshape images to (batch, input_size)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # forward pass
                outputs = resnet(images)
                val_loss = self.criterion(outputs, labels)
                # print(outputs.data.shape, type(outputs.data))
                # print(outputs.shape, type(outputs))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print("current accuracy in each batch size is {}".format(100 * correct / total ))
            val_accuracy = 100 * correct / total 
            self.val_loss_history.append(val_loss)
            self.max_loss = max(self.max_loss, val_loss)
            self.min_loss = min(self.min_loss, val_loss)
        return val_accuracy
    
    def test(self):
        resnet = torch.load(self.best_model_name).to(self.device)
        # Test model
        resnet.eval()

        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(self.test_loader):
                #  reshape images to (batch, input_size)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # forward pass
                outputs = resnet(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print("current accuracy in each batch size is {}".format(100 * correct / total ))

    def plot_curve(self): 

        title = 'the accuracy curve of train/validate'

        dpi = 80  
        width, height = 1200, 800
        legend_fontsize = 10
        scale_distance = 48.8
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.epochs)]) # epochs
        y_axis = np.zeros(self.epochs)

        plt.xlim(0, self.epochs)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.epochs + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)
      
        y_axis = self.train_accuracy_history
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train_accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis = self.val_accuracy_history
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid_accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        fig_name = self.save_history_path + "/resnet_accuracy_history"
        fig.savefig(fig_name, dpi=dpi, bbox_inches='tight')
        print ('---- save accuracy history figure {} into {}'.format(title, self.save_history_path))
        plt.close(fig)
        
        
        title = 'the loss curve of train/validate'
        fig = plt.figure(figsize=figsize)
        
        x_axis = np.array([i for i in range(self.epochs)]) # epochs
        y_axis = np.zeros(self.epochs)

        plt.xlim(0, self.epochs)
        plt.ylim(self.min_loss, self.max_loss)
        interval_y = (self.max_loss - self.min_loss) / 20
        interval_x = 5
        plt.xticks(np.arange(0, self.epochs + interval_x, interval_x))
        plt.yticks(np.arange(max(0,self.min_loss - 5 * interval_y), self.max_loss + 2 * interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('loss', fontsize=16)
        
        y_axis = self.train_loss_history
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train_loss', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis = self.val_loss_history
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='val_loss', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)
        
        fig_name = self.save_history_path + "/resnet_loss_history"
        fig.savefig(fig_name, dpi=dpi, bbox_inches='tight')
        print ('---- save loss_history figure {} into {}'.format(title, self.save_history_path))
        plt.close(fig)