import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import Config

class LossFun(nn.Module):
    def __init__(self):
        super(LossFun,self).__init__()

    def forward(self, prediction,targets,priors_boxes):
        loc_data , conf_data = prediction
        # loc_data [1840 * (4 * 4 or 6 * 4)] 
        # conf_data [1840 * (4 * class_num or 6 * class_num)]
        loc_data = torch.cat([o.view(o.size(0),-1,4) for o in loc_data] ,1) # [n, 8732, 4], n为batch，每次用多少张图片训练
        conf_data = torch.cat([o.view(o.size(0),-1,Config.class_num) for o in conf_data],1) # [n, 8732, num_class]
        priors_boxes = torch.cat([o.view(-1,4) for o in priors_boxes],0) # [8732, 4]
        if Config.use_cuda:
            loc_data = loc_data.cuda()
            conf_data = conf_data.cuda()
            priors_boxes = priors_boxes.cuda()
        # batch_size
        batch_num = loc_data.size(0)
        # default_box数量
        box_num = loc_data.size(1)

        # 存储targets根据每一个prior_box变换后的数据
        target_loc = torch.Tensor(batch_num,box_num,4) # [batch_size, 8732, 4]
        target_loc.requires_grad_(requires_grad=False)
        # 存储每一个default_box预测的种类
        target_conf = torch.LongTensor(batch_num,box_num) # [batch_size, 8732]
        target_conf.requires_grad_(requires_grad=False)
        if Config.use_cuda:
            target_loc = target_loc.cuda()
            target_conf = target_conf.cuda()
        # 因为一次batch可能有多个图，每次循环计算出一个图中的box，即8732个box的loc和conf，存放在target_loc和target_conf中
        for batch_id in range(batch_num):
            target_truths = targets[batch_id][:,:-1].data # ground_truth [1, n_boxes, 4]
            target_labels = targets[batch_id][:,-1].data
            if Config.use_cuda:
                target_truths = target_truths.cuda()
                target_labels = target_labels.cuda()
            # 计算box函数，即公式中loc损失函数的计算公式
            utils.match(0.5,target_truths,priors_boxes,target_labels,target_loc,target_conf,batch_id)
            # target_loc, target_conf => 这里实际上返回的是target_label_new, the size same as loc_data

        pos = target_conf > 0 # [batch_size, 8732] => 类别大于0的是正类，这里相当于做了一个mask
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data) # [batch_size, 8732, 4] # 选出置信度大于0的预测框
        # 相当于论文中L1损失函数乘xij的操作
        pre_loc_xij = loc_data[pos_idx].view(-1,4) # [batch * 8732, 4]
        tar_loc_xij = target_loc[pos_idx].view(-1,4)

        # 将计算好的loc和预测进行smooth_li损失函数
        loss_loc = F.smooth_l1_loss(pre_loc_xij,tar_loc_xij,size_average=False)


        batch_conf = conf_data.view(-1, Config.class_num) #  [batch * 8732, n_classs=21]

        # 参照论文中conf计算方式，求出ci
        loss_c = utils.log_sum_exp(batch_conf) - batch_conf.gather(1, target_conf.view(-1, 1)) # (n * 8732, 21)
        # target_conf.view(-1, 1) => [batch * 8732, 1] # 由预测框选出最可能对应的实际框，取出实际框的label，在预测值里面找出此真实label对应的得分
        # gather 沿给定轴dim，将输入索引张量index指定位置的值进行聚合。
        # target_conf.view(-1, 1) => (n * 8732, 1)

        loss_c = loss_c.view(batch_num, -1) # [batch_size, 8732]
        # 将正样本设定为0
        loss_c[pos] = 0

        # 将剩下的负样本排序，选出目标数量的负样本
        _, loss_idx = loss_c.sort(1, descending=True) # 将每个batch_size中的loss降序排列
        _, idx_rank = loss_idx.sort(1) # [batch_size, 8732], 将loss_idx按照升序排序，得到每个loss_index的重要性，即index_rank[number_bbox]为第number个bbox的重要性

        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(3*num_pos, max=pos.size(1)-1)

        # 提取出正负样本
        neg = idx_rank < num_neg.expand_as(idx_rank) # 选出重要性小于num_neg的框，rank从0开始，越小越重要 [0,0,1,1,....,1,0]，这可以决定选第几个框 [n, 8732]
        pos_idx = pos.unsqueeze(2).expand_as(conf_data) # [n, 8732, num_class], 这是我们要选的正类框的位置
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, Config.class_num) # [n * 8732, num_class]
        targets_weighted = target_conf[(pos+neg).gt(0)] # [n * 8732, 1] # 输出张量。必须为ByteTensor，即是否大于0。

        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False) # 其实这里的多类的crossEntropyLoss,对应着使用对数似然损失函数+softmax激活， 而不是交叉熵损失函数+sigmoid激活

        N = num_pos.data.sum().double()
        loss_l = loss_loc.double()
        loss_c = loss_c.double()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
