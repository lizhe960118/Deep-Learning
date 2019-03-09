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
### resnet
通过跳过连接解决梯度消失的问题  
训练：python main.py --model resnet --mode train  
测试：python main.py --model resnet --mode test  
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
### senet
对通道加权


## model
对于每个model，先定义它的网络层结构，然后定义一个类实现，类中会有_train,_test方法，用于模型的训练和测试。

  

  

