## 论文整理

|        简介        |                           论文题目                           |      会议       |                       链接                       |   时间   |
| :----------------: | :----------------------------------------------------------: | :-------------: | :----------------------------------------------: | :------: |
|   **CornerNet**    |      CornerNet: Keypoint Triplets for Object Detection       | **[arXiv' 18]** | [`[pdf\]`](https://arxiv.org/pdf/1808.01244.pdf) | **1808** |
|   **ExtremeNet**   | Bottom-up Object Detection by Grouping Extreme and Center Points | **[arXiv' 19]** | [`[pdf\`](https://arxiv.org/pdf/1901.08043.pdf)  | **1901** |
| **CornerNet-Lite** |  CornerNet-Lite: Efficient Keypoint Based Object Detection   | **[arXiv' 19]** | [`[pdf\]`](https://arxiv.org/pdf/1904.08900.pdf) | **1904** |
|  **Segmentation**  |                Segmentations is All You Need                 | **[arXiv' 19]** | [`[pdf\]`](https://arxiv.org/pdf/1904.13300.pdf) | **1904** |
|     **Fovea**      |        FoveaBox: Beyond Anchor-based Object Detector         | **[arXiv' 19]** | [`[pdf\]`](https://arxiv.org/pdf/1904.03797.pdf) | **1904** |
|  **CenterNet^1**   |                      Objects as Points                       | **[arXiv' 19]** | [`[pdf\]`](https://arxiv.org/pdf/1904.07850.pdf) | **1904** |
|  **CenterNet^2**   |      CenterNet: Keypoint Triplets for Object Detection       | **[arXiv' 19]** | [`[pdf\]`](https://arxiv.org/pdf/1904.08189.pdf) | **1904** |
|     **DuBox**      | DuBox: No-Prior Box Objection Detection via Residual Dual Scale Detectors | **[arXiv' 19]** | [`[pdf\]`](https://arxiv.org/pdf/1904.06883.pdf) | **1904** |
|   **RepPoints**    |   RepPoints: Point Set Representation for Object Detection   | **[arXiv' 19]** | [`[pdf\]`](https://arxiv.org/pdf/1904.11490.pdf) | **1904** |
|      **FSAF**      | Feature Selective Anchor-Free Module for Single-Shot Object Detection | **[arXiv' 19]** | [`[pdf\]`](https://arxiv.org/pdf/1903.00621.pdf) | **1903** |

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

