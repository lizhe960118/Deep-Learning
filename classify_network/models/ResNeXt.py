#ResNeXt将ResNet中的操作重复基数多次，从而增加参数的利用率

# 原来是 28 * 28 * 256
# 引入基数32，压缩率为2， 使用32个平行的卷积层，使其输出通道为4
# 之后32个 28 * 28 * 4 的特征图进行3 * 3卷积
# 最后 28 * 28 * 4的特征图进行 1*1卷积，输出 28 * 28 * 256
# 将32个特征图相加，加上残差，得到输出
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

class ResNeXtBottleBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, cardinality=16, stride=1, downsample=None):
        super(ResNeXtBottleBlock, self).__init__()

        block_channel = in_channel // (cardinality * self.expansion)
        
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channel, block_channel * cardinality, kernel_size=1),
            nn.BatchNorm2d(block_channel * cardinality),
            nn.ReLU()
            )
        self.conv_conv = nn.Sequential(
            nn.Conv2d(block_channel * cardinality, block_channel * cardinality, kernel_size=3, stride=stride,padding=1, groups=cardinality),
            # groups: 控制输入和输出之间的连接，
            # group=1，输出是所有的输入的卷积；
            # group=2，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。
            nn.BatchNorm2d(block_channel * cardinality),
            nn.ReLU()
            )
        self.conv_expand = nn.Sequential(
            nn.Conv2d(block_channel * cardinality, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
            )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv_reduce(x)
        out = self.conv_conv(out)
        out = self.conv_expand(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = out + residual
        return out 

class ResNeXt(nn.Module):
    def __init__(self, block, depth, cardinality, num_classes):
        super(ResNeXt,self).__init__()

        n_blocks = int((depth - 2) // 9)
        # 计算残差块的数量
        self.cardinality = cardinality

        # 输入 32 * 32 * 3
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            # 32 * 32 * 3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            )

        self.in_channel = 64
        # 传入需要重复的基础块， 传入需要输出的通道数， 传入基础块需要循环的次数
        self.layer1 = self._make_layer(block, 64, n_blocks)
        # 32 * 32 * 64 => 32 * 32 * 64

        self.layer2 = self._make_layer(block, 128, n_blocks, stride = 2)
        # 输出 16 * 16 * 128
        self.layer3 = self._make_layer(block, 256, n_blocks, stride = 2)
        # 输出 8 * 8 * 256

        self.avgpool = nn.AvgPool2d(8)
        # 输出256

        self.fc = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.ReLU(),
            nn.Softmax()
            ) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_channel, n_blocks, stride=1):
        # out_channel 需要输出的通道数，blocks 需要叠加几次block
        downsample = None
        if stride != 1 or self.in_channel != out_channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel, 
                    kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
                )

        layers = []

        layers.append(block(self.in_channel, out_channel, self.cardinality, stride, downsample))
        
        self.in_channel = out_channel

        for i in range(1, n_blocks):
            layers.append(block(self.in_channel, out_channel, self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out 


# def resnext29_16(num_classes=10):
#     model = ResNeXt(ResNeXtBottleBlock, depth=29, cardinality=16, num_classes=num_classes)
#     return model 

class resnext_model(object):

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
        self.train_data, self.test_data = load_data(self.dataset_path, net_name="resnext")

        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        self.trian_batch_nums = len(self.train_loader) // self.batch_size
        self.test_batch_nums = len(self.test_loader) // self.batch_size
        
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []
        self.cur_model_name = os.path.join(self.save_model_path, 'current_resnext_net.t7')
        self.best_model_name = os.path.join(self.save_model_path, 'best_resnext_net.t7')
        self.max_loss = 0
        self.min_loss = float("inf")
        
    # def adjust_learning_rate(self, epoch, optimizer):
    #     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #     self.learning_rate *= (0.1 ** (epoch // 30))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = self.learning_rate

    def train(self):
        resnext = ResNeXt(ResNeXtBottleBlock, depth=29, cardinality=16, num_classes=self.num_classes).to(self.device)
        optimizer = Adam(resnext.parameters(), betas=(.5, 0.999), lr=self.learning_rate)
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
                outputs = resnext(images)
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
            torch.save(resnext, self.cur_model_name)
            
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
        resnext = torch.load(self.cur_model_name).to(self.device)
        # Test model
        resnext.eval()

        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(self.test_loader):
                #  reshape images to (batch, input_size)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # forward pass
                outputs = resnext(images)
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
        resnext = torch.load(self.best_model_name).to(self.device)
        # Test model
        resnext.eval()

        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(self.test_loader):
                #  reshape images to (batch, input_size)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # forward pass
                outputs = resnext(images)
                
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

        fig_name = self.save_history_path + "/resnext_accuracy_history"
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
        
        fig_name = self.save_history_path + "/resnext_loss_history"
        fig.savefig(fig_name, dpi=dpi, bbox_inches='tight')
        print ('---- save loss_history figure {} into {}'.format(title, self.save_history_path))
        plt.close(fig)