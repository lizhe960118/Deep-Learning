import numpy as np
import torch
import torch.nn as nn

class ROIPool(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size)
        self.size = output_size

    def forward(self, images, rois, roi_idx):
        # images：特征图 image_batchsize * channels * h * w
        # rois:[[x1,y1,x2,y2], ...] n * 4
        # roi_idx:[4,5,8,7] n * 1, roi_idx[i]保存的是rois[i]对应的是哪个特征图


        n = rois.shape[0] # 有多少个建议框
        h = images.size(2)
        w = images.size(3)
        x1 = rois[:,0] # 提取框的位置，此处缩放为到（0,1）
        y1 = rois[:,1]
        x2 = rois[:,2]
        y2 = rois[:,3]

        x1 = np.floor(x1 * w).astype(int) # 回归到特征图的位置
        x2 = np.ceil(x2 * w).astype(int)
        y1 = np.floor(y1 * h).astype(int)
        y2 = np.ceil(y2 * h).astype(int)
        
        res = []
        for i in range(n):
            img = images[roi_idx[i]].unsqueeze(0)
            img = img[:, :, y1[i]:y2[i], x1[i]:x2[i]]
            img = self.maxpool(img) # 调用的self.maxpool直接输出output_size*output_size大小的特征图
            res.append(img)
        res = torch.cat(res, dim=0) # n * output_size * output_size
        return res
