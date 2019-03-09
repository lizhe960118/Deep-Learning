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

"""
VGG 的优点在于，堆叠多个小的卷积核而不使用池化操作可以增加网络的表征深度，同时限制参数的数量。
例如，通过堆叠 3 个 3×3 卷积层而不是使用单个的 7×7 层，可以克服一些限制。
首先，这样做组合了三个非线性函数，而不只是一个，使得决策函数更有判别力和表征能力。
第二，参数量减少了 81%，而感受野保持不变。
另外，小卷积核的使用也扮演了正则化器的角色，并提高了不同卷积核的有效性。

VGG 的缺点在于，其评估的开销比浅层网络更加昂贵，内存和参数（140M）也更多。
这些参数的大部分都可以归因于第一个全连接层。
结果表明，这些层可以在不降低性能的情况下移除，同时显著减少了必要参数的数量。
"""

class VGG16_Net(nn.Module):
    def __init__(self, num_classes):
        super(VGG16_Net, self).__init__()

        self.layer1 = nn.Sequential(
            # 核心思想用小卷积核代替大卷积核，减少参数
            # 卷积 1-1
            # 输入 （224 * 224 * 3） 卷积（3 * 3 * 3）步长 1,填充1， 个数 64
            # 输出 （224 - 3 + 2 * 1）/ 1 + 1 = 224 （224 * 224 * 64）
            # 归一化BN
            # relu处理 （relu提高训练速度）
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 卷积 1-2
            # 输入 （224 * 224 * 64） 卷积（3 * 3 * 3）步长 1,填充1， 个数 64
            # 输出 （224 - 3 + 2 * 1）/ 1 + 1 = 224 （224 * 224 * 64）
            # 归一化BN
            # relu处理
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # maxpool - 1 
            # kernel_size = 2, 步长2
            # （224 * 224 * 64）=> )(112, 112, 64)
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.layer2 = nn.Sequential(
            # 112 * 112 * 64 => 112 * 112 * 128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # conv 2 -2
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # maxpool (112 * 112 * 128) => (56 * 56 *128)
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer3 = nn.Sequential(
            # (56 * 56 * 128) = (56 * 56 * 256)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # conv 3-2
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # conv 3-3
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # max_pool (56 * 56 *256 => 28 * 28 * 256)
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer4 = nn.Sequential(
            # (28 * 28 * 256) = (28 * 28 * 512)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # conv 4-2
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # conv 4 -3        
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # max_pool (28 * 28 * 512 => 14 * 14 * 512)
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer5 = nn.Sequential(
            # (14 * 14 * 512) => (14 * 14 * 512)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # conv 5-2
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # conv 5-3        
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # max_pool (14 * 14 * 512 => 7 * 7 * 512)
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.layer6 = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU())
        self.layer8 = nn.Sequential(
            nn.Linear(4096, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.Softmax())


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        return out 

# vgg16 = VGG16_Net()

class vgg16net_model(object):

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
        self.train_data, self.test_data = load_data(self.dataset_path, net_name="vgg16net")

        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        self.trian_batch_nums = len(self.train_loader) // self.batch_size
        self.test_batch_nums = len(self.test_loader) // self.batch_size
        
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []
        self.cur_model_name = os.path.join(self.save_model_path, 'current_vgg16_net.t7')
        self.best_model_name = os.path.join(self.save_model_path, 'best_vgg16_net.t7')
        self.max_loss = 0
        self.min_loss = float("inf")
        
    # def adjust_learning_rate(self, epoch, optimizer):
    #     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #     self.learning_rate *= (0.1 ** (epoch // 30))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = self.learning_rate

    def train(self):
        
        vgg16_net = VGG16_Net(self.num_classes).to(self.device)
        optimizer = Adam(vgg16_net.parameters(), betas=(.5, 0.999), lr=self.learning_rate)
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
                outputs = vgg16_net(images)
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
            torch.save(vgg16_net, self.cur_model_name)
            
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
        vgg16_net = torch.load(self.cur_model_name).to(self.device)
        # Test model
        vgg16_net.eval()

        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(self.test_loader):
                #  reshape images to (batch, input_size)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # forward pass
                outputs = vgg16_net(images)
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
        vgg16_net = torch.load(self.best_model_name).to(self.device)
        # Test model
        vgg16_net.eval()

        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(self.test_loader):
                #  reshape images to (batch, input_size)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # forward pass
                outputs = vgg16_net(images)
                
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

        fig_name = self.save_history_path + "/accuracy_history"
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
        print(interval_y)
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
        
        fig_name = self.save_history_path + "/loss_history"
        fig.savefig(fig_name, dpi=dpi, bbox_inches='tight')
        print ('---- save loss_history figure {} into {}'.format(title, self.save_history_path))
        plt.close(fig)