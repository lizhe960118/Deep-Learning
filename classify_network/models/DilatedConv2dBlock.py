# 空洞卷积 dilated convolutional
# https://www.zhihu.com/question/54149221/answer/192025860
# 在标准的卷积核中注入空洞然后和特征图卷积
# dilation rate：卷积核中每两个数间填充0的个数 - 1 

# 在图像分割FCN中，有两个关键
# 一是pooling 减小图像尺寸增大感受野
# 二是upsampling 扩大图片尺寸，但是这个过程中肯定有一些信息损失掉了

# 普通卷积 kernel_size = k ,stride = s, padding = p 
# 感受野的计算 f_out = (f_in - k + 2p) // s + 1
# 则有 f_in = (f_out - 1) * s + k - 2p

# k, s=1, p=0 时， f_in = f_out - 1 + k => f_in - f_out = (k - 1)
# 迭代:
# f_out = 1, f_in = k = 1 * (k - 1) + 1
# f_out_1 = k, f_in_2 = k - 1 + k = 2k - 1 = 2 * (k - 1) + 1 
# 则对应着之前的n层的感受野为： (kernel_size - 1) * n + 1 (和层数呈线性关系)

# k, s, p=0时，
# f_in = (f_out - 1) * s + k
# f_out_0 = 1, f_in_1 = k 
# f_out_1 = k ,f_in_2 = (k - 1) * s + k = (k - 1) * s + (k - 1) + 1
# f_out_2 = (k - 1) * s + k, f_in_2 = (k-1) *s^2 + (k - 1) * s + (k - 1) + 1
# 则：f_in_n = (k - 1) * s^n + (k - 1) * s^(n - 1) + ... + (k - 1) + 1 


# 标准卷积 f_in = (f_out - 1) * s + k - 2padding

# 空洞卷积 k_real = k_dilated + (k_dilated - 1) * (d - 1) = d * (k_dilated - 1) + 1
# 3 * 3 卷积，dilation=2 => 3 + 2 * (2 - 1) => 5 * 5 标准卷积 => 2 * 2 + 1
# 3 * 3 卷积，dilation=4 => 3 + 2 * (4 - 1) => 9 * 9 标准卷积 => 2 * 4 + 1
# 3 * 3 卷积，dilation=8 => 3 + 2 * (8 - 1) => 17 * 17 标准卷积 => 2 * 8 + 1
# 3 * 3 卷积，dilation=16 => 3 + 2 * (16 - 1) => 33 * 33 标准卷积 => 2 * 16 + 1

# f_in = (f_out - 1) * s + d * (k_dilated - 1) + 1 - 2 * padding
# 令 f_in = f_out, s = 1, 则
# 标准卷积： k_real = 2 * padding + 1
# 空洞卷积： d * (k_dilated  - 1)  + 1 = 2 * padding + 1
# d = 2 * padding // (k_dilated  - 1)

# 两种算法：
# 正向算法：如果知道前一层感受野 f_(n-1)，则f_n = f_(n-1) + k - 1 (k为标准kernel_size)
# 后向算法：假设当前感受野为1（f_now), 则 f_prev = f_now + k_now - 1 (k为标准kernel_size)

# 空洞卷积 k_real = d * (k_dilated - 1) + 1
import torch
import torch.nn as nn

input_tensor = torch.arange(1, 26, dtype=torch.float).reshape(1, 1, 5, 5)
print(input_tensor)

norm_conv = nn.Conv2d(1, 1, 3, stride=1, dilation=1)
nn.init.constant_(norm_conv.weight, 1)

print(norm_conv(input_tensor).detach().numpy())
# torch.Size([1, 1, 7, 7])

dilated_conv = nn.Conv2d(1, 1, 3, stride=1, dilation=2)
nn.init.constant_(dilated_conv.weight, 1)
print(dilated_conv(input_tensor).detach().numpy())
# torch.Size([1, 1, 5, 5])