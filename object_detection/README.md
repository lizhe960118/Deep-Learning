##YOLO

### yolo-v1
将识别问题转化为分类问题，将原始图片分成7x7的块，每个块对应两个框和一种分类，每个框有（x,y,w,h,confidence)5个参数，分类有20个通道，总共（5 * 2 + 20）=30个通道，神经网络输出7x7x30的张量。
之前对训练图片进行处理，每个图片得到target_tensor,神经网络得到的pred_tensor，二者计算loss。
在预测时，神经网络得到predict_tensor,要将此张量转化为图片中的标注框及其对应类别和预测概率。
特定：预测的框的中心一定在此单元格内，预测的x加上单元格的位置就可以得到预测框中心的坐标。

### yolo-v2
改进：
- 使用BN（批处理正则化）
- 使用高分辨率的图像训练
- 加入anchor机制，feature map每个像素点对应特定anchors
- 使用kmeans聚类来找到模板框（piror）的大小
- 直接预测相对于网格单元左上角的相对位置（RNN预测相对于锚框中心的位置）

### yolo-v3
- 使用leaky Relu 作为激活函数
- darknet53
- 上采样然后concat，类似于FPN

##ssd（单图像多目标框检测）

- 多尺度预测：使用大小不同的特征检测不同尺度的目标
- 多种宽高比的default box
- 数据增强：放大:每个patch的大小为原图大小的[0.1, 1],宽高比在1/2,到2之间；缩小：创建16倍与原图大小的画布，将原图放置其中

##R-CNN(region-based cnn)

通过selective search选出候选框，然后对应到原图片中，对crop(裁剪)后的图片resize到同样大小通过cnn进行分类
##SPP-Net(Spatial Pyramid Pooling) 空间金字塔池化

实现了将输入任意尺度的特征图组合成特定维度的输出
##fast r-cnn

输入图片，cnn提取特征之后的到m*m的特征图，通过selective search选出候选框，对应到卷积提取过的特征图中,得到(WxH)的特征图，对这个特征图做ROIpooling（adaptivePooling），得到固定大小的输出特征，在此基础上确定回归和分类
##faster rcnn

特点：预测的tx是预测框相对当前锚框的缩放偏移量
在第五个卷积层后加入RegionProposalNetwork，用于提取建议框，对于卷积后特征图上的每一个点（共S=hh*ww个），分配A个锚框。

rpn中:
卷积层经过3*3卷积集中信息，之后使用1x1卷积构造想输入的特征
hh, ww是特征图的大小，n,channel, hh, ww = x.shape;
位置卷积output: [n, 4 * A, hh, ww];
resize=>[n, S*A,4] => rpn_loc [SxA, 4]直接得到
分类卷积output: [n, 2 * A, hh, ww];  => rpn_cls_score[S*A,2]直接得到
anchor_base => anchor 直接得到

AnchorTargetCreator:
输入所有锚框anchor，真实框bbox，输出gt_rpn_loc, gt_rpn_score,
首先由anchor和bbox算出对每个anchor由最大IOU的bbox，依次为基准gt_bbox，根据IOU的分类规则，（max_iou大于0.7的1, max_iou<0.3的为0，是某一目标框的最大IOU锚框设置为1,设置采样数256，正类比率0.5限制所选的框数量）生成大小为[SxA]的label，并给label做上标记，以此得到gt_rpn_label，由anchor和gt_bbox可以得到gt_rpn_loc

计算rpn_loss:
rpn_loc, gt_rpn_loc, rpn_cls_scores, gt_rpn_label
loc之间用smoothL1损失函数，分类之间用crossEntropyLoss

rpn_fg_socre = rpn_cls_score[:..,:1] 前景得分（生成）
ProposalCreator：输入rpn_loc，rpn_fg_score，anchor
输出：rois(这里用到nms）,第一次得到的建议框
ProposalTargetCreator:
输入rois[R',4]，bbox[R, 4],label[R,1] 即(真实框对应的label)，
输出sample_rois[S,4], gt_roi_loc[S,4], gt_roi_label[S,1]
sample_rois是第二次得到的建议框，框数量128，正类比率0.25， max_iou>0.5为选择的有物体类，maxiou<0.5,>=0为选取无物体类（背景）。得到每个simple_roi,及找到其对应的gt_bbox，可得到gt_roi_loc

sample_rois,image_size,feature输入到ROIPooling层中，每个roi得到固定输出，[S, roipoolsize**2],经过全连接层4096,4096，对应设置输出21， 21*4，得到，roi_cls_loc[S, 21x4],根据gt_roi_label,选出对应层的loc,即为roi_loc；roi_scores[S, 21],将与gt_roi_label计算交叉熵损失

### FPN（特征金字塔网络）
同时利用低层特征高分辨率和高层的语义信息，通过融合这些不同的层的特征达到预测的结果，预测是在每个融合后的特征层上单独进行的。

### FocalLoss（聚焦损失函数）
Focal loss 主要是为了解决one-stage目标检测中正负样本比例严重失衡的问题，该损失函数降低了大量简单负样本在训练中所占的权重。样本越易分，pt越大，则贡献的loss就越小.（这里之后就不像two_stage算法一样做正负样本的均衡）
[Focal loss](https://www.cnblogs.com/king-lps/p/9497836.html)

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
- NMS（NonMaximumSuppresion) 非极大值抑制（选出置信度最高的框后，将与他相交的IOU大于阈值的框去掉

### 评价标准
- mAP（mean Average Precision)
用预测框和实际框计算出的IOU值与设定的IOU阈值比较，可以计算出某张图像中某个类C的正确检测次数A;对于每个图像，我们知道该图像中给定类别C的实际目标的数量B.则该类C的精度, P=(A/B)
有100张图像，可得到100个精度值，这100个精度的平均值，得到的是该类C的平均精度，AP=(sum(P)/N_images)
取所有类的平均精度值的平均，衡量模型的性能：
mAP=（sum(AP)/ N_class)
- FPS (Frame Per Second)
1秒内识别的图像帧数