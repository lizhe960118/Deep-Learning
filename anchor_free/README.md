## 论文整理

|        简介        |                           论文题目                           |      会议       |                      链接                       |   时间   |
| :----------------: | :----------------------------------------------------------: | :-------------: | :---------------------------------------------: | :------: |
|   **CornerNet**    |      CornerNet: Keypoint Triplets for Object Detection       | **[arXiv' 18]** | [`[pdf]`](https://arxiv.org/pdf/1808.01244.pdf) | **1808** |
|   **ExtremeNet**   | Bottom-up Object Detection by Grouping Extreme and Center Points | **[arXiv' 19]** | [`[pdf]`](https://arxiv.org/pdf/1901.08043.pdf) | **1901** |
| **CornerNet-Lite** |  CornerNet-Lite: Efficient Keypoint Based Object Detection   | **[arXiv' 19]** | [`[pdf]`](https://arxiv.org/pdf/1904.08900.pdf) | **1904** |
|  **Segmentation**  |                Segmentations is All You Need                 | **[arXiv' 19]** | [`[pdf]`](https://arxiv.org/pdf/1904.13300.pdf) | **1904** |
|     **Fovea**      |        FoveaBox: Beyond Anchor-based Object Detector         | **[arXiv' 19]** | [`[pdf]`](https://arxiv.org/pdf/1904.03797.pdf) | **1904** |
|  **CenterNet^1**   |                      Objects as Points                       | **[arXiv' 19]** | [`[pdf]`](https://arxiv.org/pdf/1904.07850.pdf) | **1904** |
|  **CenterNet^2**   |      CenterNet: Keypoint Triplets for Object Detection       | **[arXiv' 19]** | [`[pdf]`](https://arxiv.org/pdf/1904.08189.pdf) | **1904** |
|     **DuBox**      | DuBox: No-Prior Box Objection Detection via Residual Dual Scale Detectors | **[arXiv' 19]** | [`[pdf]`](https://arxiv.org/pdf/1904.06883.pdf) | **1904** |
|   **RepPoints**    |   RepPoints: Point Set Representation for Object Detection   | **[arXiv' 19]** | [`[pdf]`](https://arxiv.org/pdf/1904.11490.pdf) | **1904** |
|      **FSAF**      | Feature Selective Anchor-Free Module for Single-Shot Object Detection | **[arXiv' 19]** | [`[pdf]`](https://arxiv.org/pdf/1903.00621.pdf) | **1903** |

##  源码链接

|        论文        |                             源码                             |
| :----------------: | :----------------------------------------------------------: |
|   **CornerNet**    | (official, Pytorch) https://github.com/princeton-vl/CornerNet |
| **CornerNet-Lite** | (official, Pytorch) https://github.com/princeton-vl/CornerNet-Lite |
|      **FCOS**      |   (official, Pytorch) https://github.com/tianzhi0549/FCOS    |
|  **CenterNet^1**   | (official, Pytorch) https://github.com/xingyizhou/CenterNet  |
|  **CenterNet^2**   | (official, pytorch) https://github.com/Duankaiwen/CenterNet  |
|   **ExtremeNet**   | (official, Pytorch) https://github.com/xingyizhou/ExtremeNet |

## 论文分析

### 预测的目标

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

### 正负样本的选定

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

##  代码学习

|      目录       |         备注          |
| :-------------: | :-------------------: |
| test_bceloss.py |     测试BCE loss      |
|      flop       | 对于不同网络计算Flops |
|    backbone     |  骨干网络的学习笔记   |

## 常见分类网络的准确度

error在imagetnet上测试出，越小越好  

params(M)是保存的模型大小, flops(G)是模型完成一次预测需要的计算量

|      Models      | top-1-error | top5-error | Mparams | Gflops |
| :--------------: | :---------: | :--------: | :-----: | :----: |
|     alexnet      |    43.45    |   20.91    |  61.10  |  0.71  |
|      vgg-16      |    28.41    |    9.62    | 138.36  | 15.62  |
|  **vgg-16_bn**   |    26.63    |    8.50    | 138.37  | 15.65  |
|  **ResNet-50**   |    23.85    |    7.13    |  25.56  |  3.53  |
|  **ResNet-101**  |    22.63    |    6.44    |  44.55  |  7.26  |
|    ResNet-152    |    21.69    |    5.94    |  60.19  | 10.99  |
|   DenseNet-121   |    25.35    |    7.83    |  7.98   |  2.79  |
|   DenseNet-161   |    22.35    |    6.20    |  28.68  |  7.69  |
| **DenseNet-169** |    24.00    |    7.00    |  14.15  |  3.33  |
|   Densenet-201   |    22.80    |    6.43    |  20.01  |  4.28  |
|  SqueezeNet 1.0  |    41.90    |   19.58    |  1.25   |  0.70  |
|  SqueezeNet 2.0  |    41.81    |   19.38    |  1.24   |  0.34  |
|   Inception v3   |    22.55    |    6.44    |         |        |

## 各种目标检测网络的flops和准确度

- **two-stage**

|              Models              |         backbone         | mini-val(mAP) | test-dev(mAP) | params-(M) | Gflops | FPS  |  S   |  M   |  L   |
| :------------------------------: | :----------------------: | :-----------: | :-----------: | :--------: | :----: | :--: | :--: | :--: | :--: |
|   Faster R_CNN+++(1000 * 600)    |      ResNet-101-C4       |               |     34.9      |            |        |  5   | 15.6 | 38.7 | 50.9 |
| Faster R_CNN w FPN  (1000 * 600) |      ResNet-101-FPN      |               |     36.2      |            |        |      | 18.2 | 39.0 | 48.2 |
|        DFCN (1000 * 600)         | Aligned-Inception-ResNet |               |     37.5      |            |        |      | 19.4 | 40.1 | 52.5 |
|     Mask R_CNN (1300 * 800)      |       ResNeXt-101        |               |     39.8      |            |        |      | 22.1 | 43.2 | 51.2 |
|      Soft_NMS (1300 * 800)       | Aligned-Inception-ResNet |               |     40.9      |            |        |      | 23.3 | 43.6 | 53.3 |
|          Cascade R_CNN           |        ResNet-101        |               |     42.8      |            |        |      | 23.7 | 45.5 | 55.2 |
|  Grid R_CNN w FPN  (1300 * 800)  |       ResNeXt-101        |               |     43.2      |            |        |      | 25.1 | 46.5 | 55.2 |

- **one-stage**

|            Models            |    backbone     | mini-val(mAP) | test-dev(mAP) | params-(M) | Gflops | FPS  |  S   |  M   |  L   |
| :--------------------------: | :-------------: | :-----------: | :-----------: | :--------: | :----: | :--: | :--: | :--: | :--: |
|            YOLOv1            |                 |               |               |            |        |  21  |      |      |      |
|      YOLOv2 (544 * 544)      |   DarkNet-19    |               |     21.6      |            |        | 78.6 | 5.0  | 22.4 | 35.5 |
|            SSD513            | ResNet-101-SSD  |               |     31.2      |            |        |  19  | 10.2 | 34.5 | 49.8 |
|           DSSD513            | ResNet-101-DSSD |               |     33.2      |            |        |      | 13.0 | 35.4 | 51.1 |
|       RetinaNet (800)        | ResNet-101-FPN  |               |     39.1      |            |        |      | 21.8 | 42.7 | 50.2 |
|          RetinaNet           | ResNeXt-101-FPN |               |     40.8      |            |        |      | 24.1 | 44.2 | 51.2 |
|          YOLOv3-608          |   Darknet-53    |               |     33.0      |            |        |  27  | 18.3 | 35.4 | 41.9 |
|    RefineDet-512 (single)    |   ResNet-101    |               |     36.4      |            |        |      | 16.6 | 39.9 | 51.4 |
|    RefineDet-512  (multi)    |   ResNet-101    |               |     41.8      |            |        |      | 25.6 | 45.1 | 54.1 |
|    CornerNet-511 (single)    |  Hourglass-52   |               |     37.8      |            |        |      | 17.0 | 39.0 | 50.2 |
|    CornerNet-511  (multi)    |  Hourglass-52   |               |     39.4      |            |        |      | 18.9 | 41.2 | 52.7 |
|    CornerNet-511 (single)    |  Hourglass-104  |               |     40.5      |            |        |      | 19.4 | 42.7 | 53.9 |
|    CornerNet-511  (multi)    |  Hourglass-104  |               |     42.1      |            |        |      | 20.8 | 44.8 | 56.7 |
| CenterNet-511 (key) (single) |  Hourglass-52   |               |     41.6      |            |        |      | 22.5 | 43.1 | 54.1 |
| CenterNet-511 (key) (multi)  |  Hourglass-52   |               |     43.5      |            |        |      | 25.3 | 45.3 | 55.0 |
| CenterNet-511 (key)(single)  |  Hourglass-104  |               |     44.9      |            |        |      | 25.6 | 47.4 | 57.4 |
| CenterNet-511 (key) (multi)  |  Hourglass-104  |               |     47.0      |            |        |      | 28.9 | 49.9 | 58.9 |