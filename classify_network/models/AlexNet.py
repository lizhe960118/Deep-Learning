# https://blog.csdn.net/zyqdragon/article/details/72353420

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

"""
# 批处理大小 128
batch_size = 128

train_dataset = torchvision.datasets.ImageFolder(root='./../data/train_data',
                                                transform = transforms.ToTensor())
test_dataset = torchvision.datasets.ImageFolder(root="./../data/train_data",
                                                transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)
"""

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        # 第一层卷积
        # 输入 （227 * 227 * 3） 卷积（11 * 11 * 3）步长 4 个数 96 
        # 输出 （227 - 11）/ 4 + 1 = 55 （55 * 55 * 96）
        # relu1处理 （relu提高训练速度）
        # 第一层池化 （pool提高精度，防止过拟合）
        # 尺度 （3 * 3）步长 2
        # 输出 （55 - 3）/ 2 + 1 = 27 (27 * 27 * 96)
        # 归一化BN
        self.layer1 = nn.Sequential(
            # input Channel, output channel, kernel_size, stride, padding
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96)
            )
        # 第二层卷积
        # 输入（27 * 27 * 96）分层两组在两个CPU中计算
        # （27 * 27 * 48）卷积（5 * 5 * 48）步长：1, 填充：2， 每组个数128
        # 输出（27 - 5 + 2 * 2）/1 + 1 = 27 （27 * 27 * 128）
        # 经过relu2处理，生成激活像素
        # 第二层池化
        # 尺度 （3 * 3），步长2
        # 输出（27 - 3）/2 + 1 = 13 (13 * 13 * 128)
        # 归一化（local_size:5 * 5）
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.BatchNorm2d(256)
            )
        # 第三次卷积
        # 输入（13 * 13 * 128）卷积（3 * 3 * 128）,步长：1，填充1，个数192
        # 输出（13 - 3 + 2*1）/1 + 1 = 13 (13 * 13 * 192)
        # relu3处理
        self.layer3 = nn.Sequential(
            nn.Conv2d(256,384, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        # 第四次卷积
        # 输入（13 * 13 * 192）卷积（3 * 3 * 192）,步长：1，填充1，个数192
        # 输出（13 - 3 + 2*1）/1 + 1 = 13 (13 * 13 * 192)
        # relu4处理
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384,kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        # 第五次卷积
        # 输入 13 * 13 * 192）卷积（3 * 3 * 192）,步长：1，填充1，个数128
        # 输出（13 - 3 + 2*1）/1 + 1 = 13 (13 * 13 * 128)
        # relu5处理
        # 池化层
        # 尺度（3 * 3），步长2
        # （13 - 3）/ 2 +1 =6 (6 * 6 * 128)
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))

        # 第五层数据合并
        # 输入（6 * 6 * 256），卷积（6 * 6 * 256），步长1，个数4096
        # 输出 （6 - 6）/ 1 + 1 = 1 (4096个神经元)
        # relu6处理
        # drop6处理
        self.fc_1 = nn.Linear(6 * 6 * 256, 4096)
        self.dropout_1 =  nn.Dropout(p=0.5)

        # 全连接层
        # 输入 4096 (4096个神经元)
        # relu7
        # drop7
        self.fc_2 = nn.Linear(4096, 4096)
        self.dropout_2 = nn.Dropout(0.6)

        # 全连接层
        # 输入 4096 （1000个神经元）
        # 输出训练值
        self.fc_3=  nn.Linear(4096, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)

        out = self.fc_1(out)
        out = F.relu(out)
        out = self.dropout_1(out)

        out = self.fc_2(out)
        out = F.relu(out)
        out = self.dropout_2(out)

        out = self.fc_3(out)
        out = F.softmax(out)
        return out 

# alex_net = AlexNet()
"""
# 用一个均值为0、标准差为0.01的高斯分布初始化了每一层的权重
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0, 0.01)
    if classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0,.0.01)
        m.bias.data.fill_(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlexNet().to(device)
model.apply(weight_init)

criterion = nn.CrossEntropyLoss()
# 当验证误差率在当前学习率下不再提高时，就将学习率除以10。
learning_rate = 0.01
# 动力0.8，权重衰减为5e-5
# eps = 1e-8, weight_decay=5e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-08, weight_decay=5e-05)

# Trian the model
total_steps = len(train_loader)
num_epochs = 1
for epoch in range(num_epochs):
    for i,(images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch:[{}/{}],Step:[{}/{}],Loss:{:.4f}'.format(epoch+1, num_epochs, step, total_steps, loss))


# test the model
model.eval()
# eval()时，框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值

with torch.no_grad():
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        predict_labels = model(images)

        _, predicted = torch.max(predict_labels, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print("Accuracy of the model on 100 test images is{}%".format(correct / total * 100))

# save the model
torch.save(model.state_dict(), 'AlexNet_model.ckpt')
"""

class alexnet_model(object):
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
        self.epochs = epochs
        self.batch_size = batchsize
        self.mode = mode
        self.learning_rate = 0.0001
        self.num_classes = 10
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda:1")
        else:
            self.device = torch.device("cpu")
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.train_data, self.test_data = load_data(self.dataset_path, net_name="alexnet")

        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        self.trian_batch_nums = len(self.train_loader) // self.batch_size
        self.test_batch_nums = len(self.test_loader) // self.batch_size
        
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []
        self.cur_model_name = os.path.join(self.save_model_path, 'current_alex_net.t7')
        self.best_model_name = os.path.join(self.save_model_path, 'best_alex_net.t7')

    def train(self):
        alex_net = AlexNet(self.num_classes).to(self.device)
        optimizer = Adam(alex_net.parameters(), lr=self.learning_rate)
        step = 0
        for epoch in range(self.epochs):
            self.adjust_learning_rate(epoch, optimizer)
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
                outputs = alex_net(images)
                train_loss = self.criterion(outputs, labels)

                # compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # compute gradient and do Adam step
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                print("epoch:{}, step:{}, loss:{}".format(epoch, step, train_loss.item()))
                
                #show result and save model 
                # step = epochs * (50000 // batch_size) = 100 * 50000 // 100 = 50000
                # if (step % 100) == 0:
            # each epoch store the history and accurate
            self.train_loss_history.append(train_loss)
            # 计算模型在训练集上准确率
            train_accuracy = 100 * correct / total 
            self.train_accuracy_history.append(train_accuracy)
            
            # 保存当前模型
            torch.save(alex_net, self.cur_model_name)
            
            # 计算当前模型在测试集上的准确率的准确率
            val_accuracy = self.validate()
            self.val_accuracy_history.append(val_accuracy)

            # 如果准确率高，则替换最佳模型为当前模型
            if val_accuracy > max(self.val_accuracy_history[:-1]):
                # 每一轮训练之后进行验证，如果平均准确率高
                # 这里加上准确率的判断，如果在验证集的前多少取得的准确率较高，则替换模型
                shutil.copyfile(self.cur_model_name, self.best_model_name) # 把cur_model复制给best_model

        # 调用print_history将loss画出来并保存为图片
        # print_accarucy将在验证集上的准确率保存下来，存到图片里
        self.plot_curve()

    def validate(self):
        alex_net = torch.load(self.cur_model_name).to(self.device)
        # Test model
        alex_net.eval()

        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(cur_data_loader):
                #  reshape images to (batch, input_size)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # forward pass
                outputs = alex_net(images)
                val_loss = self.criterion(outputs, labels)
                # print(outputs.data.shape, type(outputs.data))
                # print(outputs.shape, type(outputs))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print("current accuracy in each batch size is {}".format(100 * correct / total ))
            val_accuracy = 100 * correct / total 
            self.val_loss_history.append(val_loss)
        return val_accuracy

    def adjust_learning_rate(self, epoch, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.learning_rate *= (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.learning_rate
        
    def plot_curve(self): 

        title = 'the loss/accuracy curve of train/validate'

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
      
        y_axis[:] = self.train_accuracy_history
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train_accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.val_accuracy_history
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid_accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        
        y_axis[:] = self.train_loss_history
        plt.plot(x_axis, y_axis*50, color='g', linestyle=':', label='train_loss', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.val_loss_history
        plt.plot(x_axis, y_axis*50, color='y', linestyle=':', label='val_loss', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        fig.savefig(self.save_history_path, dpi=dpi, bbox_inches='tight')
        print ('---- save figure {} into {}'.format(title, self.save_history_path))
        plt.close(fig)
