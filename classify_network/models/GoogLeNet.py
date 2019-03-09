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

class inception_module(nn.Module):
    def __init__(self, in_channel, out_channels):
        super(inception_module, self).__init__()

        """
        # 1x1 convolution
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channels[0], kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU()
            )

        # bottleneck layer and 3x3 convolution
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channels[1], kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels[1]),
            nn.ReLU(),
            nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[2]),
            nn.ReLU()
            )

        # bottleneck layer and 5x5 convolution
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(in_channel, out_channels[3], kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(),
            nn.Conv2d(out_channels[3], out_channels[4], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channels[4]),
            nn.ReLU()
            )
        # maxpooling and bottleneck layer

        self.maxpool_3x3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channels[5], kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels[5]),
            nn.ReLU()
            )
        """
        # 1x1 convolution
        self.conv_1x1 = nn.Conv2d(in_channel, out_channels[0], kernel_size=1, stride=1)

        # bottleneck layer and 3x3 convolution
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channels[1], kernel_size=1, stride=1),
            nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, stride=1, padding=1)
            )

        # bottleneck layer and 5x5 convolution
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(in_channel, out_channels[3], kernel_size=1, stride=1),
            nn.Conv2d(out_channels[3], out_channels[4], kernel_size=5, stride=1, padding=2)
            )

        # maxpooling and bottleneck layer
        self.maxpool_3x3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel, out_channels[5], kernel_size=1, stride=1)
            )

    def forward(self, x):
        conv_1x1_out = self.conv_1x1(x)
        conv_3x3_out = self.conv_3x3(x)
        conv_5x5_out = self.conv_5x5(x)
        maxpool_3x3_out = self.maxpool_3x3(x)
        out = torch.concat([conv_1x1_out, conv_3x3_out, conv_5x5_out, maxpool_3x3_out], 1)
        return out 

class aux_classifier(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(aux_classifier, self).__init__()
        # Average pooling, fc, dropout, fc
        self.average_pool = nn.AvePool2d(kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv2d(in_channel, 128, kernel_size=1, stride=1)
        # self.fc_1 = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(p=0.7)
        self.fc_2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.average_pool(x)
        out = self.conv(out)
        out = out.reshape(out.size(0), -1)

        self.fc_1 = nn.Linear(out.size(-1),1024)

        out = self.fc_1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc_2(out)
        out = F.softmax(out)
        return out 

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()

        # stage 1 - layers before inception modules
        # 输入 32 * 32 * 3
        self.stage_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 16 * 16 * 64
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            # 16 * 16 * 192
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # 16 * 16 * 192
            )

        # stage2 - 2 inception modules and max pooling
        self.stage_2_1 = inception_module(192, out_channels=[64, 96, 128, 16, 32, 32])
        # 16 * 16 * 256
        self.stage_2_2 = inception_module(256, out_channels=[128, 128, 192, 32, 96, 64])
        # 16 * 16 *  480
        self.stage_2_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 8 * 8 *  480

        # stage3 - 5 inception modules and max pooling
        self.stage_3_1 = inception_module(480, out_channels = [192, 96, 208, 16, 48, 64])
        # 8 * 8 * 512
        self.stage_3_aux_1 = aux_classifier(512, num_classes)

        self.stage_3_2 = inception_module(512, out_channels = [160, 112, 225, 24, 64, 64])
        self.stage_3_3 = inception_module(512, out_channels = [128, 128, 256, 24, 64, 64])
        # 8 * 8 * 512
        self.stage_3_4 = inception_module(528, out_channels = [112, 144, 288, 32, 64, 64])
        # 8 * 8 * 528
        self.stage_3_aux_2 = aux_classifier(528, num_classes)

        self.stage_3_5 = inception_module(528, out_channels = [256, 160, 320, 32, 128, 128])
        # 8 * 8 * 832
        self.stage_3_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 4 * 4 * 832

        # stage4 - 2 inception modules and average pooling
        self.stage_4_1 = inception_module(832, out_channels = [256, 160, 320, 32, 128, 128])
        # 4 * 4 * 832
        self.stage_4_2 = inception_module(832, out_channels = [384, 192, 384, 48, 128, 128])
        # 4 * 4 * 1024
        self.stage_4_avgpool = nn.AvgPool(kernel_size=4, stride=1)
        # 1 * 1 * 1024

        # stage5 - dropout,linear fc, softmax fc
        self.stage_5_drop = nn.Dropout(0.4)
        self.stage_5_fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.stage_1(x)

        out = self.stage_2_1(out)
        out = self.stage_2_2(out)
        out = self.stage_2_maxpool(out)

        out = self.stage_3_1(out)
        auc_output_1 = self.stage_3_aux_1(out)
        out = self.stage_3_2(out)
        out = self.stage_3_3(out)
        out = self.stage_3_4(out)
        auc_output_2 = self.stage_3_aux_2(out)
        out = self.stage_3_5(out)
        out = self.stage_3_maxpool(out)

        out = self.stage_4_1(out)
        out = self.stage_4_2(out)
        out = self.stage_4_avgpool(out)

        out = out.reshape(out.size(0), -1)
        out = self.stage_5_drop(out)
        out = self.stage_5_fc(out)
        out = F.softmax(out)

        return [out, auc_output_1, auc_output_2]

class googlenet_model(object):

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
        self.train_data, self.test_data = load_data(self.dataset_path, net_name="googlenet")

        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        self.trian_batch_nums = len(self.train_loader) // self.batch_size
        self.test_batch_nums = len(self.test_loader) // self.batch_size
        
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []
        self.cur_model_name = os.path.join(self.save_model_path, 'current_google_net_net.t7')
        self.best_model_name = os.path.join(self.save_model_path, 'best_google_net_net.t7')
        self.max_loss = 0
        self.min_loss = float("inf")
        
    # def adjust_learning_rate(self, epoch, optimizer):
    #     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #     self.learning_rate *= (0.1 ** (epoch // 30))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = self.learning_rate

    def train(self):
        google_net = GoogLeNet(num_classes=10).to(device).to(self.device)
        optimizer = Adam(google_net.parameters(), betas=(.5, 0.999), lr=self.learning_rate)
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
                outputs = google_net(images)
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
            torch.save(google_net, self.cur_model_name)
            
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
        google_net = torch.load(self.cur_model_name).to(self.device)
        # Test model
        google_net.eval()

        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(self.test_loader):
                #  reshape images to (batch, input_size)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # forward pass
                outputs = google_net(images)
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
        google_net = torch.load(self.best_model_name).to(self.device)
        # Test model
        google_net.eval()

        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(self.test_loader):
                #  reshape images to (batch, input_size)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # forward pass
                outputs = google_net(images)
                
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

        fig_name = self.save_history_path + "/googlenet_accuracy_history"
        fig.savefig(fig_name, dpi=dpi, bbox_inches='tight')
        print ('---- save accuracy history figure {} into {}'.format(title, self.save_history_path))
        plt.close(fig)
        
        
        title = 'the accuracy curve of train/validate'
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
        
        fig_name = self.save_history_path + "/googlenet_loss_history"
        fig.savefig(fig_name, dpi=dpi, bbox_inches='tight')
        print ('---- save loss_history figure {} into {}'.format(title, self.save_history_path))
        plt.close(fig)

