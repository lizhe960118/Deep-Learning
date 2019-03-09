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

# dense块是一个residual的极端版本，其中每个卷积层都会和这个块之前所有卷积块的输出相连。
# https://blog.csdn.net/u013841196/article/details/80725764
class BottleneckBlock(nn.Module):
    expansion = 4

    # 这里只进行同大小的卷积，不进行缩放
    def __init__(self, in_channel, growthRate):
        # out_channel = growthRate
        # growthRate表示每个denseBlock中每层输出的featureMap个数
        super(BottleneckBlock, self).__init__()
        # 1 * 1 卷积缩小输出通道 
        # 例如第32层（31 * growthRate) =>(growthRate * 4)
        # BN-ReLU-Conv
        self.conv_layer1 = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channel, growthRate // self.expansion, kernel_size=1),
            )

        # 这里每个dense block的3*3卷积前面都包含了一个1*1的卷积操作，就是所谓的bottleneck layer
        # 3 * 3 卷积
        self.conv_layer2 = nn.Sequential(
            nn.BatchNorm2d(growthRate // self.expansion),
            nn.ReLU(inplace = True),
            nn.Conv2d(growthRate // self.expansion, growthRate, kernel_size=3, padding=1),
            )

        # # 1 * 1 卷积恢复
        # self.conv_layer3 = nn.Sequential(
        #     nn.Conv2d(out_channel // self.expansion, out_channel, kernel_size=1),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU(inplace = True)
        #     )

    def forward(self, x):
        residual = x
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        # 在channel上将他们拼接起来
        out = torch.cat((residual, out), 1)
        return out 

class BasicBlock(nn.Module):
    # 未使用1 * 1卷积降维， 只经过一次 3*3 卷积之后， 原来的输入图片拼接
    def __init__(self, in_channel, growthRate):
        super(SingleLayer, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channel,growthRate, kernel_size=3, padding=1),
            )

    def forward(self, x):
        residual = x
        out = self.conv_layer(x)
        out = torch.cat((residual, out), 1)
        return out 

class TransitionLayer(nn.Module):
    # 将之前所有concat的层做一次1*1卷积降维

    # 虽然第32层的3*3卷积输出channel只有32个（growth rate），但是紧接着还会像前面几层一样有通道的concat操作，
    # 即将第32层的输出和第32层的输入做concat，前面说过第32层的输入是1000左右的channel，
    # 所以最后每个Dense Block的输出也是1000多的channel。
    # 因此这个transition layer有个参数reduction（范围是0到1），表示将这些输出缩小到原来的多少倍，默认是0.5，
    # 这样传给下一个Dense Block的时候channel数量就会减少一半，这就是transition layer的作用。

    # 使之输出通道数为out_channel = in_channel * reduction
    # 最后做一次平均池化,输出通道数
    def __init__(self, in_channel, out_channel):
        super(TransitionLayer, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1,stride=1),
            nn.AvgPool2d(kernel_size=2,stride=2)
            )
    def forward(self, x):
        out = self.conv_layer(x)
        return out
    

class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, num_classes, useBottleneck):
        super(DenseNet, self).__init__()
        if useBottleneck:
            # 计算 dense_block的数量 
            # （卷积conv1，pooling1）, TransitionLayer(3), （pool_last, fc_last）
            # depth = 5 + 3 * 2 * n_blocks = 65
            n_blocks = (depth - 5) // 2
            
        else:
            n_blocks = (depth - 5) // 1

        # 3个denseBlocks, 计算每个denseBlock中的bottleneck/basicBlock的个数
        n_blocks = n_blocks // 3
        # 假设每一个denseblock有10个bottleneck

        # 输入 224 * 224 * 3
        in_channel = 2 * growthRate # in_channel = 64

        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, in_channel, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        # 56 * 56 * 64

        self.dense1 = self._make_dense(in_channel, growthRate, n_blocks, useBottleneck)
        in_channel += growthRate * n_blocks
        # 56 * 56 * (64 + 32 * 10) = 56 * 56 * 384

        out_channel = int(in_channel * reduction)
        self.trans1 = TransitionLayer(in_channel, out_channel)
        # 56 * 56 * 384 = > 28 * 28 * 192
        in_channel = out_channel # in_channel = 192

        self.dense2 = self._make_dense(in_channel, growthRate, n_blocks, useBottleneck)
        in_channel += growthRate * n_blocks
        # 28 * 28 * 192 => 28 * 28 * (192 + 320) => 28 * 28 * 512

        out_channel = int(in_channel * reduction)
        self.trans2 = TransitionLayer(in_channel, out_channel)
        # 28 * 28 * 512 => 14 * 14 * 256
        in_channel = out_channel

        self.dense3 = self._make_dense(in_channel, growthRate, n_blocks, useBottleneck)
        in_channel += growthRate * n_blocks
        # 14 * 14 * 256 => 14 * 14 * (256 + 320) = 14 * 14 * 576

        out_channel = int(in_channel * reduction) # out_channel // 2 =288
        self.trans3 = TransitionLayer(in_channel, out_channel)
        # 14 * 14 * 576 => 7 * 7 * 288
        in_channel = out_channel #  288

        self.avg_pool = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.AvgPool2d(7)
            )
        
        self.fc = nn.Linear(in_channel, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, in_channel, growthRate, n_blocks, useBottleneck):
        layers = []

        for i in range(int(n_blocks)):
            if useBottleneck:
                layers.append(BottleneckBlock(in_channel, growthRate))
            else:
                layers.append(BasicBlock(in_channel, growthRate))
            in_channel += growthRate

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.avg_pool(out)
        out = out.reshape(out.size(0),-1)
        out = self.fc(out)
        return out 

# def densenet65_32(num_classes=10):
#     model = dense_net(32, 65, 0.5, num_classes=num_classes, useBottleneck=True)
#     return model

# densenet65_32 = densenet65_32()

# DenseNet核心思想在于建立了不同层之间的连接关系，充分利用了feature，进一步减轻了梯度消失问题，加深网络不是问题，而且训练效果非常好。
# 另外，利用bottleneck layer，Translation layer以及较小的growth rate使得网络变窄，参数减少，有效抑制了过拟合，同时计算量也减少了。

"""
def dense_block(self, x, f=32, d=5):
    l = x
    for i in range(d):
        x = conv(l, f)
        l = concatnate([l, x])
    return l
"""
class densenet_model(object):

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
        self.train_data, self.test_data = load_data(self.dataset_path, net_name="densenet")

        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        self.trian_batch_nums = len(self.train_loader) // self.batch_size
        self.test_batch_nums = len(self.test_loader) // self.batch_size
        
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []
        self.cur_model_name = os.path.join(self.save_model_path, 'current_densenet_net.t7')
        self.best_model_name = os.path.join(self.save_model_path, 'best_densenet_net.t7')
        self.max_loss = 0
        self.min_loss = float("inf")
        
    # def adjust_learning_rate(self, epoch, optimizer):
    #     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #     self.learning_rate *= (0.1 ** (epoch // 30))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = self.learning_rate

    def train(self):
        dense_net = DenseNet(32, 65, 0.5, num_classes=self.num_classes, useBottleneck=True).to(self.device)
        optimizer = Adam(dense_net.parameters(), betas=(.5, 0.999), lr=self.learning_rate)
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
                outputs = dense_net(images)
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
            torch.save(dense_net, self.cur_model_name)
            
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
        dense_net = torch.load(self.cur_model_name).to(self.device)
        # Test model
        dense_net.eval()

        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(self.test_loader):
                #  reshape images to (batch, input_size)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # forward pass
                outputs = dense_net(images)
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
        dense_net = torch.load(self.best_model_name).to(self.device)
        # Test model
        dense_net.eval()

        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(self.test_loader):
                #  reshape images to (batch, input_size)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # forward pass
                outputs = dense_net(images)
                
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

        fig_name = self.save_history_path + "/densenet_accuracy_history"
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
        
        fig_name = self.save_history_path + "/densenet_loss_history"
        fig.savefig(fig_name, dpi=dpi, bbox_inches='tight')
        print ('---- save loss_history figure {} into {}'.format(title, self.save_history_path))
        plt.close(fig)