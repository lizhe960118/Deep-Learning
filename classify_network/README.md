# 分类网络

## 数据集
cifar-10：共10种类型,共60000张32*32*3的图片，其中训练集50000张，测试集10000张。  
下载地址：http://www.cs.toronto.edu/~kriz/cifar.html 
下载完成解压缩会得到：data_batch1~data_batch5,每个文件包含10000张图片，test_batch包含10000张图片。

## data_utils.py
loda_data:对数据集进行读取
check_folder:文件夹不存在则创建文件夹

## main.py
传入要网络名称和模式，  
如：python main.py --model alexnet --mode train为训练alexnet。  
python main.py --model alexnet --mode test 为测试最佳alex网络的性能。

## model文件夹
包含各种分类网络
### AlexNet
基础的分类网络   
训练：python main.py --model alexnet --mode train  
测试：python main.py --model alexnet --mode test 

### VGG16
使用3x3小卷积代替7x7卷积，减少网络参数    

训练：python main.py --model vgg16net --mode train  
测试：python main.py --model vgg16net --mode test  

(堆叠多个小的卷积核而不使用池化操作可以增加网络的表征深度，同时限制参数的数量)

### resnet
通过跳过连接解决梯度消失的问题  

训练：python main.py --model resnet --mode train  
测试：python main.py --model resnet --mode test  

(增加了神经网络架构的跳过连接（skip connection），使用批归一化并移除了作为最后一层的全连接层)

### resnet_bottleneck
resnet中基础块换成bottleneck
训练：python main.py --model resnet_bottleneck --mode train  
测试：python main.py --model resnet_bottleneck --mode test
### resnext
使用多个并行的resnet基础块，分别卷积后求和，再和残差相加输出   
训练：python main.py --model resnext --mode train  
测试：python main.py --model resnetx --mode test
### densenet
resnet升级版，当前层和之间的每一层都有连接，这里是concatnate  
训练：python main.py --model densenet --mode train  
测试：python main.py --model densenet --mode test
### googlenet
引入inception机制，1x1卷积，(1x1 + 3x3卷积), （1x1 + 5x5卷积）,(3x3 maxpool + 1x1 conv),stride都为1，out_channel自定义，最后concat。有多余两个aux_classifer，用来回传梯度。  
训练：python main.py --model googlenet --mode train    
测试：python main.py --model googlenet --mode test  

(通过构建由多个子模块组成的复杂卷积核来提高卷积核的学习能力和抽象能力)

### senet
对通道加权
训练：python main.py --model senet --mode train 
测试: python main.py --model senet --mode test
### csrnet
使用了空洞卷积
### Xception
https://blog.csdn.net/u014380165/article/details/75142710

###inception v2

增加正则化，5x5改成两个3x3

###Inception V3

inception V3把googlenet里一些7*7的卷积变成了1*7和7*1的两层串联，3*3的也一样，变成了1*3和3*1，
这样加速了计算，还增加了网络的非线性，减小过拟合的概率。另外，网络的输入从224改成了299.
使用小卷积，尝试使用非对称卷积通过执行批归一化和标签平滑化来改进正则化。标签平滑就是为每个类都分配一些权重，而不是将全权重分配给 ground truth 标签。

###inception V4

inception v4实际上是把原来的inception加上了resnet的方法，从一个节点能够跳过一些节点直接连入之后的一些节点，并且残差也跟着过去一个。 另外就是V4把一个先1*1再3*3那步换成了先3*3再1*1。论文说引入resnet不是用来提高深度，进而提高准确度的，只是用来提高速度的。

## model

对于每个model，先定义它的网络层结构，然后定义一个类实现，类中会有_train,_test方法，用于模型的训练和测试。

## debug
### 训练loss不下降
调小learning_rate或者batch_size
### 训练准确率上不去
注意参数的初始化

  

  

