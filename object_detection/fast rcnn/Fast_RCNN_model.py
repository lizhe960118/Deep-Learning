import torch
import torch.nn
import torchvision
from torch.autograd import Variable
from roi import *

class FastRCNN(nn.module):
    def __init__(self, n_class):
        super(RCNN, self).__init__()
        rawnet = torchvision.models.vgg16_bn(pretrained=True)
        # 得到提取特征的卷积层
        self.seq = nn.Sequential(*list(rawnet.features.children())[:-1])
        self.roipool = ROIPool(output_size=(7,7))
        self.feature = nn.Sequential(*list(rawnet.classifier.children())[:-1])

        _x = Variable(torch.Tensor(1, 3, 224, 224))
        _r = np.array([[0., 0., 1., 1.]])
        _ri = np.array([0])
        _x = self.feature(self.roipool(self.seq(_x), _r, _ri).view(1, -1))
        # 得到最后的全连接层的输入特征维数
        feature_dim = _x.size(1)

        self.cls_classifier = nn.Linear(feature_dim, (n_class + 1))
        self.bbox_regressor = nn.Linear(feature_dim, 4 * (n_class + 1))

        self.lossCls = nn.CrossEntropyLoss()
        self.smoothL1 = nn.SmoothL1Loss()

    def forward(self, images, rois, roi_idx):
        # N * 224 *224
        out = self.seq(images)
        out = self.roipool(out, rois, roi_idx)
        out = out.detach()
        out = out.view(out.size(0), -1)
        feat = self.feature(out) # 得到每个框的输出特征 n_boxes * feature_dim

        cls_score = self.cls_classifier(feat) # (n_boxes, K + 1)
        bbox_regression = self.bbox_regressor(feat).view(-1, n_class + 1, 4) #(n_boxes, K+1, 4)
        return cls_score, bbox_regression

    def fast_rcnn_loss(self, cls_score, bbox_regression, labels, gt_bbox):
        """
        n = n_boxes
        cls_score:(n, K+1)
        labels:(n) => crossEntropyLoss 会将其转化为onehot =》 (n , K + 1)
        bbox_regression:(n, K+1, 4)
        gt_bbox:(n, 4)
        """
        # 计算分类误差
        loss_cls = self.LossCls(cls_score, labels)

        labels_expand = labels.view(-1, 1, 1).expand(labels.size(0), 1, 4) # (N, 1, 4)
        # 通过label去除背景
        mask = (labels != 0).float().view(-1, 1).expand(labels.size(0), 4) # (N, 4)
        # 以labels_expand的第1维(即labels的类别)为index，在bbox_regression中选出这个类对应算出的回归框端点
        # （n, K + 1, 4) = > (n, 4)
        loss_loc = self.smoothL1(bbox_regression.gather(1, labels_expand).squeeze(1) * mask - gt_bbox * mask)

        lambda_loc = 1
        total_loss = loss_cls + lambda_loc * loss_loc
        return total_loss, loss_cls, loss_loc






