#encoding:utf-8
#
#created by xiongzihua
#
import torch
from torch.autograd import Variable
import torch.nn as nn

from net import vgg16, vgg16_bn
from resnet_yolo import resnet50
import torchvision.transforms as transforms
import cv2
import numpy as np

VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
'sheep', 'sofa', 'train', 'tvmonitor')

Color = [[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]]

def decoder(pred):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    # 将输出的 7 * 7 的特征图转换为框的位置和类别
    grid_num = 7 # 7 * 7
    boxes=[] # 框的位置
    cls_indexs=[] # 类别
    probs = [] # 概率
    cell_size = 1./grid_num
    pred = pred.data # tenser转化为numpy
    pred = pred.squeeze(0) #7x7x30

    contain1 = pred[:,:,4].unsqueeze(2) # 将 通道4取出来并拓展1维， 7 * 7 * 1（confidence）
    contain2 = pred[:,:,9].unsqueeze(2)

    contain = torch.cat((contain1,contain2),2) # 7 * 7 * 2 (0表示来自通道4， 1表示来自通道9)

    mask1 = contain > 0.1 #大于阈值 # 确定有物体
    mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9 # 选取两个通道所有块中置信度最大的块 
    mask = (mask1+mask2).gt(0) # 有的话置1 # 也就是说对于每个块，取通道4或者通道9置信率大于0.1的那个块

    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i,j,b] == 1: # 这个块某一通道置信度大于0.1
                    #print(i,j,b)
                    box = pred[i,j,b*5:b*5+4] # 这个块第一个框或者第二个框 1 * 1 * 5
                    contain_prob = torch.FloatTensor([pred[i,j,b*5+4]]) # 当前框的置信率

                    xy = torch.FloatTensor([j,i]) * cell_size #cell左上角  up left of cell （块的左上角
                    box[:2] = box[:2]*cell_size + xy # return cxcy relative to image ，这里求得中心点坐标
                    #box[:2]保存的是[delta_x, delta_y], 即[cx, cy]是框中心点距离左上角的大小， center_x = block_up_left_x + cx
                    
                    box_xy = torch.FloatTensor(box.size()) # 转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    # box_xy求得回归框两端点坐标

                    max_prob,cls_index = torch.max(pred[i,j,10:],0) # 从后面的通道取得分类的概率和类别

                    if float((contain_prob * max_prob)[0]) > 0.1: # 预测概率= 置信率 * 分类概率
                        boxes.append(box_xy.view(1,4))
                        cls_indexs.append(cls_index)
                        probs.append(contain_prob*max_prob)

    if len(boxes) ==0:
        boxes = torch.zeros((1,4)) # tensor([[ 0.,  0.,  0.,  0.]])
        probs = torch.zeros(1) # tensor([ 0.])
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes,0) #(n,4)
        probs = torch.cat(probs,0) #(n,)
        cls_indexs = torch.cat(cls_indexs,0) #(n,)
    
    keep = nms(boxes,probs)

    return boxes[keep],cls_indexs[keep],probs[keep]

def nms(bboxes,scores,threshold=0.5):
    '''
    Non-Maximum Suppression，NMS 非极大值抑制，消除重叠的框
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0] 
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _,order = scores.sort(0,descending=True) # 按照预测概率分类
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break
        #  计算其他所有框与当前框的iou
        # 这里计算iou需要的（x1,x2,y1,y2)
        # 左上角
        xx1 = x1[order[1:]].clamp(min=x1[i]) # clamp截断， 最小值为x1[i], 当小于x1[i]时，设置为x1[i] 
        yy1 = y1[order[1:]].clamp(min=y1[i])
        # 右下角
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        # xx1,xx2等都是数组

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留IoU小于阈值的box的idx(此时是（0，2，3,..) 比之前的order小1
        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]

    return torch.LongTensor(keep)
#
#start predict one image
#
def predict_gpu(model,image_name,root_path=''):

    result = []
    image = cv2.imread(root_path+image_name)
    h,w,_ = image.shape
    img = cv2.resize(image,(448,448))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    mean = (123,117,104)#RGB
    img = img - np.array(mean,dtype=np.float32)

    transform = transforms.Compose([transforms.ToTensor(),])
    img = transform(img)
    img = Variable(img[None,:,:,:],volatile=True)
    img = img.cuda()

    pred = model(img) #1x7x7x30
    pred = pred.cpu()
    boxes, cls_indexs, probs =  decoder(pred)

    for i,box in enumerate(boxes):
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index) # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1,y1),(x2,y2),VOC_CLASSES[cls_index],image_name,prob])

    return result
        



if __name__ == '__main__':
    model = resnet50()
    print('load model...')
    model.load_state_dict(torch.load('best.pth'))
    model.eval()
    model.cuda()
    image_name = 'dog.jpg'
    image = cv2.imread(image_name)
    print('predicting...')
    result = predict_gpu(model,image_name)

    for left_up, right_bottom ,class_name, _, prob in result:
        # 选取标记框的颜色
        color = Color[VOC_CLASSES.index(class_name)]
        # 画出矩形
        cv2.rectangle(image,left_up,right_bottom,color,2)
        # 设置标签，概率值保存到小数点后面两位
        label = class_name+str(round(prob,2))

        # 设置label的大小
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1]- text_size[1]) 

        # 画出要显示label的矩形框的大小，# 设置label的起点，终点
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1) # -1 表示填充
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

    cv2.imwrite('result.jpg',image)




