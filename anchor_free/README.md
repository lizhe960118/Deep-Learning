#anchor free

##论文整理

- 



##论文分析

###预测的目标

- 对feature map上的每一个点，预测该点到识别框四个边框的距离
  - FCOS （l\*, t\*, r\*, b\*)
  - Dubox ($\Delta$w1,  $\Delta$w2, $\Delta$h1, $\Delta$h2)
  - FSAF
  - FoveaBox
- 对feature map上的每一个点，预测以该点为中心的框的宽和高
  - GA_RPN (w, h)
- CornerNet 将bbox的两个角作为关键点
  - 分别对 Top-Left， Bottom-Right heatmap上的每一个点, 预测该点为角点的概率
  - 对应的Embedding层对同类角点进行分组，两个角点确定一个框
- CenterNet : Keypoint Triplets for Object Detection
  - 基于CornerNet改进，预测三个点：两个角点和一个中心点
  - 引入center精确约束目标框，设计中心区域的定义
- Object as Points: 预测中心点和尺寸
- ExtremeNet 检测所有目标的 最上，最下，最左，最右，中心点

###正负样本的选定

- 以实际矩形框的中心(x, y),确定中心区域(0.2w,0.2h)，忽略区域，负样本区域(0.5w, 0.5h)~(w, h)
  - Guided Anchoring
  - FSAF
  - FoveaBox
- 以距离框中心0.2r的圆域为正样本区域，其他样本为负，训练时加入特殊方法将一部分区域ignore
  - DenseBox
  - DuBox
- 在框内为正样本，不在框内为负
  - FCOS
- cornernet检测关键点
  - 关键点为1，在半径r内使用高斯方程进行填充



## 代码学习

|      目录       |         备注          |
| :-------------: | :-------------------: |
| test_bceloss.py |     测试BCE loss      |
|      flop       | 对于不同网络计算Flops |
|    backbone     |  骨干网络的学习笔记   |

