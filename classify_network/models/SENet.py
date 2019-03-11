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

# 使用 Squeeze and Excitation块，这个块提出每个通道的重要性，并对他们进行加权

# 输入的是一个已经卷积过的块
class SEModule(nn.Module):
    def __init__(self, in_channel, reduction):
        super(SEModule, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1) # 1 * 1 * in_channel

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1),
            # 1 * 1 * in_channel//reduction 减少参数
            nn.BatchNorm2d(in_channel // reduction),
            nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1), # 1 * 1 * in_channel
            nn.Sigmoid(),
            )

    def forward(self, x):
        module_input = x
        out = self.avgpool(x)
        out = self.layer1(out)
        out = self.layer2(out)
        return module_input * out

class SEBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, reduction=16):
        # 当 in_channel = out_channel 时，立刻可以直接相加
        super(SEBottleneckBlock, self).__init__()
        # 1 * 1 卷积 缩小想输出的维度
        # 这里的out_channel为输出的out_channel 的1/4
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // self.expansion, kernel_size=1),
            nn.BatchNorm2d(out_channel // self.expansion))
        
        self.relu = nn.ReLU(inplace=True)

        # 特征图可能大小的改变发生在这里
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channel // self.expansion, out_channel // self.expansion, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel // self.expansion))

        # 1 * 1 、卷积还原通道数
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channel // self.expansion, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel)
            )

        self.se_module = SEModule(out_channel, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        residual = x 

        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        out = self.relu(out)

        out = self.se_module(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out 

class SENet(nn.Module):
    def __init__(self, block, layers, num_classes=10, reduction=16):

        # 输入 224 * 224 * 3
        super(SENet,self).__init__()

        self.in_channel = 64

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            # (224 - 7 + 3 * 2) // 2 + 1 = 112.5
            # 舍弃 ？
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # (112 - 3 + 2 * 1)// 2 + 1 = 57?
            # 56 * 56 * 64
            )

        # 传入瓶颈块走一下流程
        # 传入需要重复的基础块， 传入需要输出的通道数， 传入基础块需要循环的次数
        self.layer1 = self._make_layer(block, 64, layers[0], reduction=reduction)
        # 第一次 x = 56 * 56 * 64, layers[0] = 3
        # f(x) => (56 - 3 + 2 * 1)/ 1 + 1 = 56 (卷积两次形状不变)， out_channel = 64
        # 输出 56 * 56 * 64 

        # 也就是说，特征图的长宽大小由stride决定，通道数由out_channel决定
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, reduction=reduction)
        # 输出 (56 - 3 + 2 * 1)// 2 + 1= 28 out_channel = 128
        # 输出 28 * 28 * 128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, reduction=reduction)
        # 输出 14 * 14 * 256
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, reduction=reduction)
        # 输出 7 * 7 * 512

        self.avgpool = nn.AvgPool2d(7)
        # 输出512
        self.fc = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.BatchNorm1d(num_classes)
#             nn.ReLU()
            ) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_channel, blocks, stride=1, reduction=16):
        # channel 需要输出的通道数，blocks 需要叠加几次block
        downsample = None
        if stride != 1 or self.in_channel != out_channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel, 
                    kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
                )
        layers = []

        layers.append(block(self.in_channel, out_channel, stride, downsample, reduction))
        
        self.in_channel = out_channel

        for i in range(1, blocks):
            layers.append(block(self.in_channel, out_channel=out_channel, reduction=reduction))

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

class senet_model(object):

    def __init__(self, dataset_path, save_model_path, save_history_path, epochs, batchsize, device, mode):
        """
        dataset_path:'./data/train_data'
        save_path:'./data_save'
        epochs:10
        bacth_size:20
        z_dim:100
        device: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        """
        self.model_name = 'senet'

        self.dataset_path = dataset_path
        self.save_model_path = save_model_path
        self.save_history_path = save_history_path
#         self.epochs = epochs
        self.epochs = 10
#         self.batch_size = batch_size
        self.batch_size = 20
        self.mode = mode
        self.learning_rate = 5e-3
        self.num_classes = 10
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda:1")
        else:
            self.device = torch.device("cpu")
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        self.train_data, self.test_data = load_data(self.dataset_path, net_name=self.model_name)

        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        self.trian_batch_nums = len(self.train_loader) // self.batch_size
        self.test_batch_nums = len(self.test_loader) // self.batch_size
        
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []
        self.cur_model_name = os.path.join(self.save_model_path, 'current_{}.t7'.format(self.model_name))
        self.best_model_name = os.path.join(self.save_model_path, 'best_{}.t7'.format(self.model_name))
        self.max_loss = 0
        self.min_loss = float("inf")
        
    # def adjust_learning_rate(self, epoch, optimizer):
    #     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #     self.learning_rate *= (0.1 ** (epoch // 30))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = self.learning_rate

    def train(self):
        try:
            model = torch.load(self.cur_model_name).to(self.device)
            print("continue train the last model")
        except FileNotFoundError:
            model = SENet(SEBottleneckBlock, layers=[3, 4, 6, 3], num_classes=self.num_classes, reduction=16).to(self.device)
        
        optimizer = Adam(model.parameters(), betas=(.5, 0.999), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=8e-4)
        
        
        for epoch in range(self.epochs):
            
#             self.adjust_learing_rate(epoch, optimizer)
            scheduler.step()
            print("Main epoch:{}".format(epoch))
            # count the accuracy when train data
            correct = 0
            total = 0
            step = 0
            for i, (images, labels) in enumerate(self.train_loader):
                
                # batch_size=100,所以step = 50000/100 = 500
                step += 1

                images = images.to(self.device)
                labels = labels.to(self.device)
                # compute output
                outputs = model(images)
                train_loss = self.criterion(outputs, labels)

                # compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # compute gradient and do Adam step
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                if step % 100 == 0:
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
            torch.save(model, self.cur_model_name)
            
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
        self.min_loss = self.min_loss.item()
        self.max_loss = self.max_loss.item()
        self.plot_curve()

    def validate(self):
        model = torch.load(self.cur_model_name).to(self.device)
        # Test model
        model.eval()

        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(self.test_loader):
                #  reshape images to (batch, input_size)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # forward pass
                outputs = model(images)
                val_loss = self.criterion(outputs, labels)
                # print(outputs.data.shape, type(outputs.data))
                # print(outputs.shape, type(outputs))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print("current accuracy in this epoch is {}".format(100 * correct / total ))
            val_accuracy = 100 * correct / total 
            self.val_loss_history.append(val_loss)
            self.max_loss = max(self.max_loss, val_loss)
            self.min_loss = min(self.min_loss, val_loss)
        return val_accuracy
    
    def test(self):
        model = torch.load(self.best_model_name).to(self.device)
        # Test model
        model.eval()

        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(self.test_loader):
                #  reshape images to (batch, input_size)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # forward pass
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print("current accuracy in this epoch is {}".format(100 * correct / total))

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

        fig_name = self.save_history_path + "{}_accuracy_history".format(self.model_name)
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
        
        fig_name = self.save_history_path + "{}_loss_history".format(self.model_name)
        fig.savefig(fig_name, dpi=dpi, bbox_inches='tight')
        print ('---- save loss_history figure {} into {}'.format(title, self.save_history_path))
        plt.close(fig)