## YOLO
### yolo-v1
将识别问题转化为分类问题，将原始图片分成7*7的块，每个块对应两个框和一种分类，每个框有（x,y,w,h,confidence)5个参数，分类有20个通道，总共（5 * 2 + 20）=30个通道，神经网络输出7*7*30的张量。
之前对训练图片进行处理，每个图片得到target_tensor,神经网络得到的pred_tensor，二者计算loss。
在预测时，神经网络得到predict_tensor,要将此张量转化为图片中的标注框及其对应类别和预测概率。

### R-CNN(region-based cnn)
通过selective search选出候选框，然后对应到原图片中，对crop(裁剪)后的图片resize到同样大小通过cnn进行分类
### fast r-cnn
输入图片，cnn提取特征之后的到m*m的特征图，通过selective search选出候选框，对应到卷积提取过的特征图中,得到(WxH)的特征图，对这个特征图做ROIpooling（adaptivePooling），得到固定大小的输出特征，在此基础上确定回归和分类
### faster rcnn
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