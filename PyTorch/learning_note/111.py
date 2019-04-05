"""
import torch
import torch.nn as nn
"""
'''
m = nn.ReflectionPad2d(2)
input = torch.arange(9, dtype=torch.float).reshape(1,1,3,3)
print(input)
print(m(input))
m = nn.ReflectionPad2d((1, 1, 2, 0))
print(m(input))
m = nn.ReflectionPad2d((1, 1, 1, 0))
# H_out = H_in + padding_top + padding_down
print(m(input))
print(m(input).size())
'''
'''
x = torch.Tensor([[1], [2], [3]])
print(x.size())
y = x.expand(3,4)
print(y)
z = x.expand_as(y)
print(z)
'''
# x = torch.Tensor([[1.3,2]])
# x = x.ceil()
# print(x)

'''
from torchvision import models
vgg = models.vgg16_bn(pretrained=True)
print(vgg)
new_state_dict = vgg.state_dict()
# print(new_state_dict)
for k in new_state_dict.keys():
    print(k)

params_dict = dict(vgg.named_parameters())
for key,value in params_dict.items():
    print(key)
'''
'''

x = torch.Tensor([[1,1,2,2],[2,2,3,3], [3,3,4,4],[4,4,5,5],[5, 5, 6, 6]])
y = torch.Tensor([[1.5,1.5,2.5,2.5],[2.5,2.5,3.5,3.5], [3.5,3.5,4.5,4.5],[4.5,4.5,5.5,5.5]])

print(x.size(0))
print(y.size(0))

# print(x[:,:2].unsqueeze(1))
print(x[:,:2].unsqueeze(1).expand(5,4,2)) # 把 x的（x1, y1）取出来然后， 然后每个复制M次
# N, 2 => N, 1, 2 => N, M, 2 # unsqueeze是在指定维度扩展

# print(y[:,:2].unsqueeze(0))
print(y[:,:2].unsqueeze(0).expand(5,4,2))
# M, 2 => 1, M, 2 => N, M, 2 # 相当于将y的所有（x1, y1)取出来，然后集体复制M次

lt = torch.max(
    x[:,:2].unsqueeze(1).expand(5,4,2),
    y[:,:2].unsqueeze(0).expand(5,4,2),
    )
print(lt)
# print(lt.size())
# 得到交集框的左端点
rb = torch.min(
    x[:,2:].unsqueeze(1).expand(5,4,2),
    y[:,2:].unsqueeze(0).expand(5,4,2),
    )
print(rb)
'''

"""
x = torch.Tensor([[1,1,2,2],[2,2,3,3], [3,3,4,4],[4,4,5,5],[5, 5, 6, 6]])
t = x[:,3] > 3
print(t)

s = t.unsqueeze(-1)
print(s)
s = s.expand_as(x)
print("expand:", s)

s = x[s]
print(s, s.size())
s = s.view(-1,4)
print(s)
# s = s[:,:].view(-1,2)
s = s[:,:].contiguous().view(-1,2)
print(s)

noo_pred_mask = torch.ByteTensor(torch.Tensor([[2, 3],[2, 3]]).size())
print(noo_pred_mask)
noo_pred_mask.zero_()
print(noo_pred_mask)
noo_pred_mask[1] = 1
print(noo_pred_mask)

t1  = torch.Tensor([[1, 1], [2, 2]]) # 2 * 2
print(t1)
t1 = t1.unsqueeze(2) # 2 * 2 * 1
print(t1)
t2  = torch.Tensor([[2, 2], [3, 3]]).unsqueeze(2)
s = torch.cat((t1,t2),2)# 2 * 2 * 2
print(s, s.size())
"""
"""
tensor([[[ 1.,  2.], # 块和块之间的互相对应
         [ 1.,  2.]],

        [[ 2.,  3.],
         [ 2.,  3.]]]) 

torch.Size([2, 2, 2])
"""
'''
print(torch.min(s,2))
"""
(tensor([[ 1.,  1.],
        [ 2.,  2.]]), 
tensor([[ 0,  0],
        [ 0,  0]]))
"""

mask1 = s > 1
print(s.max())
"""
tensor(3.)
"""
print(mask1)
"""
tensor([[[ 0,  1],
         [ 0,  1]],

        [[ 1,  1],
         [ 1,  1]]], dtype=torch.uint8)
"""
mask2 = (s == s.max())
print(mask2)
"""
tensor([[[ 0,  0],
         [ 0,  0]],

        [[ 0,  1],
         [ 0,  1]]], dtype=torch.uint8)
"""
print(mask1 + mask1)
print((mask1+ mask1).gt(0))

print(torch.zeros(1))

scores = torch.Tensor([[1], [2], [4], [3]])
_,order = scores.sort(0,descending=True)
print(order)
"""
tensor([[ 2],
        [ 3],
        [ 1],
        [ 0]])
"""
print(order[1:])
"""
tensor([[ 3],
        [ 1],
        [ 0]])
"""
i = order[0]
# print(scores[order[1:]].clamp(min=scores[i]))
import numpy as np
a = torch.from_numpy(np.array([1, 2, 3]).astype(np.float32))
print(a)
print(a.type())
"""
'''


"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

data1 =  autograd.Variable(torch.FloatTensor([[0.1, 0.9], [0.2, 0.8],[0.3, 0.7]]))

data = autograd.Variable(torch.FloatTensor([1.0,2.0,3.0]))
log_softmax=F.log_softmax(data,dim=0)
print(log_softmax)
# tensor([-2.4076, -1.4076, -0.4076])
softmax=F.softmax(data,dim=0)
print(softmax)
# tensor([ 0.0900,  0.2447,  0.6652])

np_softmax=softmax.data.numpy()
log_np_softmax=np.log(np_softmax)
print(log_np_softmax)
# [-2.407606   -1.4076059  -0.40760598]

data = np.asarray([[1.0,2.0,3.0], [2, 3, 4]])
print(data)
print(data.argmax(axis=1))
# [2 2]
print(data.max(axis=1))
# [3. 4.]
print(data.argmax(axis=0))
# [1 1 1]
print(data.max(axis=0))
# [2. 3. 4.]

data = np.asarray([[6, 9,0,4], [11, 2, 3, 4], [7, 12, 3, 4], [3, 7, 13, 14] ])
gt_argmax_data = data.argmax(axis=0)
print(gt_argmax_data)
# [1 2 3 3]
gt_max_data = data[gt_argmax_data, np.arange(data.shape[1])]
print(gt_max_data)
# [11 12 13 14]
tmp = np.where(data == gt_max_data)
print(tmp)
# (array([1, 2, 3, 3], dtype=int64), array([0, 1, 2, 3], dtype=int64))
tmp = tmp[0]
print(tmp)
# [1 2 3 3]
# print(np.concatenate((np.asarray([[1.0,2.0,3.0]]), np.asarray([[2, 3, 4]])), axis=0).shape)
inside_index = [0, 1, 2, 3]
label = np.empty((len(inside_index),), dtype=np.int32)
label.fill(-1)
print(label)

gt_argmax_ious = [3, 2, 2]
label[gt_argmax_ious] = 1
print(label)

data = np.asarray([3, 7, 13, 14])
print(data.shape[1:])
# (4,)
"""
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

data1 =  autograd.Variable(torch.FloatTensor([[0.1, 0.9], [0.2, 0.8],[0.3, 0.7]]))
data2 =  autograd.Variable(torch.FloatTensor([1, 0, 1])).long()
print(data1, data1.shape, data1.type())
print(data2, data2.shape, data2.type())
# tensor([[ 0.1000,  0.9000],
#         [ 0.2000,  0.8000],
#         [ 0.3000,  0.7000]]) torch.Size([3, 2]) torch.FloatTensor
# tensor([ 1,  0,  1]) torch.Size([3]) torch.LongTensor

# Input: (N,C) where C = number of classes
# Target: (N) where each value is 0≤targets[i]≤C−1
cls_loss = F.cross_entropy(data1,data2)
print(cls_loss)

data3 =  autograd.Variable(torch.FloatTensor([[0.9,0.1], [0.8,0.2], [0.3,0.7]]))
# data3 = autograd.Variable(torch.FloatTensor([[0], [0], [0]])).long()

data3_0 = torch.max(data3, 1)[0]
print(data3_0, data3_0.type())
# tensor([ 0.9000,  0.8000,  0.7000]) torch.FloatTensor

data3_1 = torch.max(data3, 1)[1]
print(data3_1, data3_1.type())
# tensor([ 0,  0,  1]) torch.LongTensor

data4 =  autograd.Variable(torch.FloatTensor([[1], [0], [1]])).long()

# data_new = data4.contiguous()
# print(data_new)
# tensor([[ 1],
#         [ 0],
#         [ 1]])

data_new = data4.squeeze(1)
print(data_new, data_new.type())
# tensor([ 1,  0,  1]) torch.LongTensor


data4_0 = torch.max(data4, 1)[0]
print(data4_0, data4_0.type())
# tensor([ 1,  0,  1]) torch.LongTensor
data4_1 = torch.max(data4, 1)[1]
print(data4_1, data4_1.type())
# tensor([ 0,  0,  0]) torch.LongTensor

print(data3, data3.type())
print(data4, data4.type())
# tensor([[ 0.9000,  0.1000],
#         [ 0.8000,  0.2000],
#         [ 0.3000,  0.7000]]) torch.FloatTensor
# tensor([[ 1],
#         [ 0],
#         [ 1]]) torch.LongTensor
cls_loss = F.cross_entropy(data3, data_new)
print(cls_loss)
"""
"""
import random 
_list = [i for i in range(8)]
print(_list)
random.shuffle(_list)
print(_list)
"""
"""
import torch
x = torch.Tensor([2, 2, 3, 4, 5, 6, 8, 8, 8]) # 每个候选框对于的最优真实框
index = torch.LongTensor([3, 2, 4, 6]) # 每个真实框对应的最优候选框
x.index_fill_(0, index, -1)
print(x)

target_conf = torch.Tensor([[0.1, 0.2, -0.1], [0.4, 0.5, -0.2], [0.1, 0.8, 0.9]])
pos = target_conf > 0 
print(pos)
# tensor([[ 1,  1,  0],
#         [ 1,  1,  0],
#         [ 1,  1,  1]], dtype=torch.uint8)
print(pos.dim())
# 2
print(pos.unsqueeze(pos.dim()))
# tensor([[[ 1],
#          [ 1],
#          [ 0]],

#         [[ 1],
#          [ 1],
#          [ 0]],

#         [[ 1],
#          [ 1],
#          [ 1]]], dtype=torch.uint8)
# print(pos.unsqueeze(pos.dim()).expand((3, 4)))

# loss_c = torch.Tensor([[0.1, 0.2, -0.1], [0.4, 0.5, -0.2], [0.1, 0.8, 0.9]])
loss_c = torch.Tensor([[0.1, 1.2, 5], [1, 0.5, -0.2], [0.1, 3, 0.9]])
_, loss_idx = loss_c.sort(1, descending=True)
print(loss_idx)
# tensor([[ 2,  1,  0],
#         [ 0,  1,  2],
#         [ 1,  2,  0]])

tensor_new, idx_rank = loss_idx.sort(1)
print(tensor_new)
print(idx_rank)
"""
"""
import threading
import time

class MyThread(threading.Thread):
    def __init__(self, arg):
        # super(MyThread, self).__init__() # 新式类继承原有方法写法
        threading.Thread.__init__(self)
        self.arg = arg

    def run(self):
        time.sleep(2)
        print(self.arg)

for i in range(10):
    thread = MyThread(i)
    print(thread.name)
    thread.start()
"""

'''
import time, threading

class Ticker(threading.Thread):
  """A very simple thread that merely blocks for :attr:`interval` and sets a
  :class:`threading.Event` when the :attr:`interval` has elapsed. It then waits
  for the caller to unset this event before looping again.

  Example use::

    t = Ticker(1.0) # make a ticker
    t.start() # start the ticker in a new thread
    try:
      while t.evt.wait(): # hang out til the time has elapsed  # Event对象wait的方法只有在内部信号为真的时候才会很快的执行并完成返回。当Event对象的内部信号标志位假时，
                                                              # 则wait方法一直等待到其为真时才返回。也就是说必须set新号标志位真
        t.evt.clear() # tell the ticker to loop again
        print time.time(), "FIRING!"
    except:
      t.stop() # tell the thread to stop
      t.join() # wait til the thread actually dies

  """
  # SIGALRM based timing proved to be unreliable on various python installs,
  # so we use a simple thread that blocks on sleep and sets a threading.Event
  # when the timer expires, it does this forever.
  def __init__(self, interval):
    super(Ticker, self).__init__()
    self.interval = interval
    self.evt = threading.Event()
    self.evt.clear() # 使用Event对象的clear（）方法可以清除Event对象内部的信号标志，即将其设为假，当使用Event的clear方法后，isSet()方法返回假
    self.should_run = threading.Event() 
    self.should_run.set() #当使用event对象的set（）方法后，isSet（）方法返回真
    # print(self.evt.wait())

  def stop(self):
    """Stop the this thread. You probably want to call :meth:`join` immediately
    afterwards
    """
    self.should_run.clear()

  def consume(self):
    was_set = self.evt.is_set()
    if was_set:
      self.evt.clear()
    return was_set

  def run(self):
    """The internal main method of this thread. Block for :attr:`interval`
    seconds before setting :attr:`Ticker.evt`

    .. warning::
      Do not call this directly!  Instead call :meth:`start`.
    """
    while self.should_run.is_set(): 
    # 使用Event的set（）方法可以设置Event对象内部的信号标志为真。
    # Event对象提供了isSet（）方法来判断其内部信号标志的状态。
    # 当使用event对象的set（）方法后，isSet（）方法返回真
      time.sleep(self.interval)
      # print("i am sleeping")
      self.evt.set()
      # print("i put the evt.isSet == True")

class TimeLimitError(Exception):
    def __init__(self, value):
        Exception.__init__()
        self.value = value

    def __str__(self):
        return self.value

t = Ticker(10) # make a ticker
t.start() # start the ticker in a new thread
count_loop = 0
try:
    # start_time = time.time()
    print(time.time())
    while t.evt.wait(): # hang out til the time has elapsed  # Event对象wait的方法只有在内部信号为真的时候才会很快的执行并完成返回。当Event对象的内部信号标志位假时，
                                                   # 则wait方法一直等待到其为真时才返回。也就是说必须set新号标志位真
        # print("i am waiting")
        t.evt.clear() # tell the ticker to loop again
        # print("i put the evt.isSet == False")
        count_loop += 1
        if count_loop > 3:
            raise TimeLimitError('Time limit exceeded')
        # end_time = time.time()

        # if end_time - start_time >= 10:
        #     print(time.time(), "FIRING!")
        #     # break
except TimeLimitError as e:
  t.stop() # tell the thread to stop
  t.join() # wait til the thread actually dies

# t = Ticker(10) # make a ticker
# t.start() # start the ticker in a new thread
# print(time.time(), "FIRING!")  
# try:
#     pass
#     # while t.evt.wait(): # hang out til the time has elapsed  # Event对象wait的方法只有在内部信号为真的时候才会很快的执行并完成返回。当Event对象的内部信号标志位假时，
#     #     print(time.time(), "FIRING!")                                                # 则wait方法一直等待到其为真时才返回。也就是说必须set新号标志位真
#     #     t.evt.clear() # tell the ticker to loop again
# except:
#   t.stop() # tell the thread to stop
#   t.join() # wait til the thread actually dies
'''

import torch
import numpy as np

# iou_pred_np = np.arange(4 * 4 * 5, dtype=np.float).reshape(4 * 4, 5, 1)
# print(iou_pred_np)

# _iou_mask = np.zeros([4 * 4, 5, 1], dtype=np.float)

# iou_penalty = 0 - iou_pred_np[iou_pred_np < 9]
# print(iou_penalty)

# _iou_mask[iou_pred_np < 9] = iou_penalty
# print(_iou_mask)

boxes = torch.from_numpy(np.arange(1 * 2, dtype=np.float).reshape(1, 2))

n_boxes = boxes.shape[0]
if n_boxes == 1:
    box_1 = boxes[0]
    boxes = []
    for i in range(3):
        boxes.append(box_1)
    boxes = torch.stack(boxes)
elif n_boxes == 2:
    box_1 = boxes[0]
    box_2 = boxes[1]
    boxes = []
    boxes.append(box_1)
    boxes.append(box_1)
    boxes.append(box_2)
    boxes = torch.stack(boxes)   

print(boxes)

# cxcy = np.arange(2 * 3, dtype=np.float).reshape(3, 2)
# cxcy = torch.from_numpy(cxcy)
# print(cxcy.shape)
# for i in range(3):
#     cx, cy = cxcy[i].numpy().astype(int).tolist()
#     print(type(cx))
#     print(cy)