#encoding:utf-8
#
#created by xiongzihua 2017.12.26
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class yoloLoss(nn.Module):
    def __init__(self,S,B,l_coord,l_noobj):
        super(yoloLoss,self).__init__()
        self.S = S # 整个图片分割成 s * s
        self.B = B # 预测多少个框
        self.l_coord = l_coord # 对是物体的块loss的加权
        self.l_noobj = l_noobj # 对非物体块loss的加权

        self.grad_num = S

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        其实不再只是求矩形a和矩形b之间的IOU，而是求box1中的所有矩形和box2中矩阵，两两之间的intersection
        Args:
          box1: (tensor) bounding boxes, sized [N,4].   [[x1,y1,x2,y2],...] 维度为N
          box2: (tensor) bounding boxes, sized [M,4].   [[x1,y1,x2,y2],...] 维度为M
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )
        # box1 中所有(x1,y1) 与box2对应的相比较， 每个取最大的值，# 得到交集框的左上端点
        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )
        # box1 中所有(x2,y2) 与 box2对应的相比较， 每个取最小的值，# 得到交集框右下端点

        wh = rb - lt  # [N,M,2] #(作差求出相交矩形的宽、高)
        wh[wh<0] = 0  # clip at 0 # 小于0说明不相交
        inter = wh[:,:,0] * wh[:,:,1]  # [N, M, 2] => [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,] #box1中矩形框的面积
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M] # 复制M次
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M] # 整体复制N次

        iou = inter / (area1 + area2 - inter) # [N, M]
        return iou


    def forward(self,pred_tensor,target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        '''
        N = pred_tensor.size()[0] # 30
        coo_mask = target_tensor[:,:,:,4] > 0 #有物体标注 [batchsize, S, S] (1 or 0)
        noo_mask = target_tensor[:,:,:,4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor) # [batchsize, S, S, 30]
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_pred = pred_tensor[coo_mask].view(-1,30) # [Block_of_has_object(1-49 in each tensor) in batch_size, 30]
        box_pred = coo_pred[:,:10].contiguous().view(-1,5) # box[x1,y1,w1,h1,c1] [Block_of_has_object * 2, 5]
        class_pred = coo_pred[:,10:]  # [Block_of_has_object, 20]
        
        coo_target = target_tensor[coo_mask].view(-1,30)
        box_target = coo_target[:,:10].contiguous().view(-1,5)
        class_target = coo_target[:,10:]

        # compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1,30) # blocks has no object in batch_size,取出pred_tensor对应通道的张量，每个块额、维度为30
        noo_target = target_tensor[noo_mask].view(-1,30)

        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size()) # [blocks has no object, 30]
        noo_pred_mask.zero_()
        noo_pred_mask[:,4] = 1 # 置信率的通道置为1
        noo_pred_mask[:,9] = 1 # 

        # 取出pred_tensor 中的置信率通道的值
        noo_pred_c = noo_pred[noo_pred_mask] #noo pred只需要计算 c 的损失 size[-1,2]
        # 取出target_tensor 中的置信率通道的值
        noo_target_c = noo_target[noo_pred_mask] # 
        nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c,size_average=False)

        #compute contain obj loss
        coo_response_mask = torch.cuda.ByteTensor(box_target.size()) # [Block_of_has_object * 2, 5] (delta_x, delta_y, w, h, c)
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()

        box_target_iou = torch.zeros(box_target.size()).cuda() # [Block_of_has_object * 2, 5] 

        for i in range(0,box_target.size()[0],2): #choose the best iou box
            
            box1 = box_pred[i:i+2] # 每次取两个框 [2, 5]
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            # grad_num must be same in encoder and decoder
            # 计算box_1中两个框的位置
            box1_xyxy[:,:2] = box1[:,:2] / self.grad_num - 0.5 * box1[:,2:4] # (x1, y1) = ((center_x - w/2),(center_y - h/2) 
            box1_xyxy[:,2:4] = box1[:,:2] / self.grad_num + 0.5 * box1[:,2:4] # (x2, y2) = ((center_x + w/2),(center_y + h/2) 

            box2 = box_target[i].view(-1,5) # 取一个框就行，因为两个框相同
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:,:2] = box2[:,:2] / self.grad_num - 0.5 * box2[:,2:4]
            box2_xyxy[:,2:4] = box2[:,:2] / self.grad_num + 0.5 * box2[:,2:4]
            iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) # [2,1]

            max_iou,max_index = iou.max(0)
            max_index = max_index.data.cuda() # 取出max_index (0 or 1)
            
            coo_response_mask[i + max_index] = 1 # 选出pred的两个框中的一个
            coo_not_response_mask[i + 1 - max_index] = 1 # 两个框中没有选择的

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i + max_index,torch.LongTensor([4]).cuda()] = (max_iou).data.cuda() # confidence对应位置填入max_iou

        box_target_iou = Variable(box_target_iou).cuda()

        #1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1,5) # 预测的置信度
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5) # 实际算得的iou

        box_target_response = box_target[coo_response_mask].view(-1,5)

        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],size_average=False)
        # 位置损失
        loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False) + F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False)
        
        #2.not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1,5)
        box_target_not_response[:,4] = 0 # 没有用到的框的置信度为0
        #not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        
        #I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4],size_average=False)

        #3.class loss
        class_loss = F.mse_loss(class_pred,class_target,size_average=False)

        return (self.l_coord * loc_loss + 2 * contain_loss + not_contain_loss + self.l_noobj * nooobj_loss + class_loss) / N




