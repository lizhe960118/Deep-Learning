import torch

input_value = torch.rand(1, 1, 7, 7)
print("feature map size is:", input_value.shape[1:])

conditional_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1)
conditional_conv_out = conditional_conv(input_value)
print("the conditional convolution output size is:", conditional_conv_out.shape[1:])

dilated_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, dilation=2)
dilated_conv_out = dilated_conv(input_value)
print("the conditional convolution output size is:", dilated_conv_out.shape[1:])

conditional_conv_2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1)
conditional_conv2_out = conditional_conv_2(input_value)
print("the conditional convolution output size is:", conditional_conv_out.shape[1:])
