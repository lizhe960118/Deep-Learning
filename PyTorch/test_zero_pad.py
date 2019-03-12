import torch
import torch.nn as nn

# 镜像填充
m = nn.ReflectionPad2d(2)
input = torch.arange(9, dtype=torch.float).reshape(1,1,3,3)
print(input)
"""
tensor([[[[ 0.,  1.,  2.],
          [ 3.,  4.,  5.],
          [ 6.,  7.,  8.]]]])
"""
print(m(input))
m = nn.ReflectionPad2d((1, 1, 2, 0))
print(m(input))
"""
tensor([[[[ 7.,  6.,  7.,  8.,  7.],
          [ 4.,  3.,  4.,  5.,  4.],
          [ 1.,  0.,  1.,  2.,  1.],
          [ 4.,  3.,  4.,  5.,  4.],
          [ 7.,  6.,  7.,  8.,  7.]]]])
"""
m = nn.ReflectionPad2d((1, 1, 1, 0))
# H_out = H_in + padding_top + padding_down
print(m(input))
print(m(input).size())
m = nn.ReflectionPad2d((1, 0, 1, 0))
print(m(input))
"""
tensor([[[[ 4.,  3.,  4.,  5.],
          [ 1.,  0.,  1.,  2.],
          [ 4.,  3.,  4.,  5.],
          [ 7.,  6.,  7.,  8.]]]])
"""

### 复制填充
m = nn.ReplicationPad2d(2)
print(m(input))
"""
tensor([[[[ 0.,  0.,  0.,  1.,  2.,  2.,  2.],
          [ 0.,  0.,  0.,  1.,  2.,  2.,  2.],
          [ 0.,  0.,  0.,  1.,  2.,  2.,  2.],
          [ 3.,  3.,  3.,  4.,  5.,  5.,  5.],
          [ 6.,  6.,  6.,  7.,  8.,  8.,  8.],
          [ 6.,  6.,  6.,  7.,  8.,  8.,  8.],
          [ 6.,  6.,  6.,  7.,  8.,  8.,  8.]]]])
"""
m = nn.ReplicationPad2d((1, 0, 1, 0))
print(m(input))
"""
tensor([[[[ 0.,  0.,  1.,  2.],
          [ 0.,  0.,  1.,  2.],
          [ 3.,  3.,  4.,  5.],
          [ 6.,  6.,  7.,  8.]]]])
"""

### ZeroPad2d 零填充

m = nn.ZeroPad2d(2)
print(m(input))
"""
tensor([[[[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
          [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
          [ 0.,  0.,  0.,  1.,  2.,  0.,  0.],
          [ 0.,  0.,  3.,  4.,  5.,  0.,  0.],
          [ 0.,  0.,  6.,  7.,  8.,  0.,  0.],
          [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
          [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]]]])
"""
m = nn.ZeroPad2d((2, 0, 1, 0))
print(m(input))
"""
tensor([[[[ 0.,  0.,  0.,  0.,  0.],
          [ 0.,  0.,  0.,  1.,  2.],
          [ 0.,  0.,  3.,  4.,  5.],
          [ 0.,  0.,  6.,  7.,  8.]]]])
"""
print(m(input).size())
"""
torch.Size([1, 1, 4, 5])
"""
m = nn.ZeroPad2d((1, 0, 2, 0))
print(m(input))
"""
tensor([[[[ 0.,  0.,  0.,  0.],
          [ 0.,  0.,  0.,  0.],
          [ 0.,  0.,  1.,  2.],
          [ 0.,  3.,  4.,  5.],
          [ 0.,  6.,  7.,  8.]]]])
"""
print(m(input).size())
"""
torch.Size([1, 1, 5, 4])
"""

figure_input = torch.arange(16, dtype=torch.float).reshape(1,1,4,4)
m1 = nn.ZeroPad2d((1, 0, 1, 0))
out = m1(figure_input)
print(out)
m2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
out = m2(out)
print(out)
out = out[:,:,1:,1:]
print(out)
"""
tensor([[[[  0.,   0.,   0.,   0.,   0.],
          [  0.,   0.,   1.,   2.,   3.],
          [  0.,   4.,   5.,   6.,   7.],
          [  0.,   8.,   9.,  10.,  11.],
          [  0.,  12.,  13.,  14.,  15.]]]])
tensor([[[[  0.,   2.,   3.],
          [  8.,  10.,  11.],
          [ 12.,  14.,  15.]]]])
tensor([[[[ 10.,  11.],
          [ 14.,  15.]]]])
"""
figure_input = torch.arange(16, dtype=torch.float).reshape(1,1,4,4)
# m1 = nn.ZeroPad2d((1, 0, 1, 0))
# out = m1(figure_input)
# print(out)
m1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
out = m1(figure_input)
print(out)
"""
tensor([[[[  5.,   7.],
          [ 13.,  15.]]]])
"""