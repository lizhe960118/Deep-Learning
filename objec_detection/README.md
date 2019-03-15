## YOLO
### yolo-v1
将识别问题转化为分类问题，将原始图片分成7*7的块，每个块对应两个框和一种分类，每个框有（x,y,w,h,confidence)5个参数，分类有20个通道，总共（5 * 2 + 20）=30个通道，神经网络输出7*7*30的张量。
之前对训练图片进行处理，每个图片得到target_tensor,神经网络得到的pred_tensor，二者计算loss。
在预测时，神经网络得到predict_tensor,要将此张量转化为图片中的标注框及其对应类别和预测概率。

### R-CNN(region-based cnn)
通过selective search选出候选框，然后对应到原图片中，对crop(裁剪)后的图片resize到同样大小通过cnn进行分类
### fast r-cnn
输入图片，cnn提取特征之后的到m*m的特征图，通过selective search选出候选框，对应到卷积提取过的特征图中,得到(WxH)的特征图，对这个特征图做ROIpooling（adaptivePooling），得到固定大小的输出特征，在此基础上确定回归和分类

### 术语
- SPP： spatial pyramid pooling(空间金字塔）  
将按照给定的几种比例构造网格图，然后对应比例放大到图片中，对于每个网格块做maxpool，得到（网格块个数）* channels的特征。  
具体程序中：input： m * m, 网格块比例（n * n),使用 kernel_size = (m / n)向上取整，stride = （m / n)向下取整的卷积
- ROI： regions of interest （感兴趣的区域，特征框）
- ROI pooling:作用类似于spp，得到固定大小的输出
input feature map： W * H
ouput feature map: pooled_w * pooled_h
将ROI作用到input特征图上，然后将提取出来的区域分成pooled_w * pooled_h 块，每一块做max_pooling
- RPN： region proposal network (区域建议网络，特征框提取网络）
- anchor: 锚  