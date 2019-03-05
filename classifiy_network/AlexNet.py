# https://blog.csdn.net/zyqdragon/article/details/72353420

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

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


class AlexNet(nn.Module):
    def __init__(self):
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
            nn.Relu(),
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
            nn.Relu(),
            nn.MaxPool2d(kernel_size=3,stride=2)
            nn.BatchNorm2d(256)
            )
        # 第三次卷积
        # 输入（13 * 13 * 128）卷积（3 * 3 * 128）,步长：1，填充1，个数192
        # 输出（13 - 3 + 2*1）/1 + 1 = 13 (13 * 13 * 192)
        # relu3处理
        self.layer3 = nn.Sequential(
            nn.Conv2d(256,384, kernel_size=3, stride=1, padding=1),
            nn.Relu())
        # 第四次卷积
        # 输入（13 * 13 * 192）卷积（3 * 3 * 192）,步长：1，填充1，个数192
        # 输出（13 - 3 + 2*1）/1 + 1 = 13 (13 * 13 * 192)
        # relu4处理
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384,kernel_size=3, stride=1, padding=1),
            nn.Relu())
        # 第五次卷积
        # 输入 13 * 13 * 192）卷积（3 * 3 * 192）,步长：1，填充1，个数128
        # 输出（13 - 3 + 2*1）/1 + 1 = 13 (13 * 13 * 128)
        # relu5处理
        # 池化层
        # 尺度（3 * 3），步长2
        # （13 - 3）/ 2 +1 =6 (6 * 6 * 128)
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.Relu(),
            nn.MaxPool2d(kernel_size=3, stride=2))

        # 第五层数据合并
        # 输入（6 * 6 * 256），卷积（6 * 6 * 256），步长1，个数4096
        # 输出 （6 - 6）/ 1 + 1 = 1 (4096个神经元)
        # relu6处理
        # drop6处理
        self.layer6 = nn.Sequential(
            nn.Linear(6 * 6 * 256, 4096),
            nn.Relu(),
            nn.Dropout2d(p=0.5))

        # 全连接层
        # 输入 4096 (4096个神经元)
        # relu7
        # drop7
        self.layer7 = nn.Sequential(
            nn.Linear(4096, 4096),
            # nn.BatchNorm1d(4096),
            nn.Relu(),
            nn.Dropout2d(0.5))

        # 全连接层
        # 输入 4096 （1000个神经元）
        # 输出训练值
        self.layer8 = nn.Sequential(
            nn.Linear(4096, 1000),
            nn.BacthNorm1d(1000),
            nn.Softmax()
            )
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