import torch
from quantizer.quant_lsq import *


# With Learnable Parameters
m = nn.BatchNorm2d(100)
# Without Learnable Parameters
m = nn.BatchNorm2d(100, affine=False)
input = torch.randn(20, 100, 35, 45)
output = m(input)

mm = QuanBatchNorm(100)
q_output = mm(input)
print(output, "======", q_output)


'''
N, C, H, W = 1, 3, 10, 10
input = torch.randn(N, C, H, W)
# Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
# as shown in the image below
layer_norm = nn.LayerNorm([C, H, W])
output = layer_norm(input)
print(output)
print("================")

q_layer_norm = QuanLayerNorm([C, H, W])
q_output = q_layer_norm(input)
print(q_output)
'''
'''
# With square kernels and equal stride
m = nn.Conv2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = torch.randn(1, 16, 50, 100)
output = m(input)
print(output)
print("===========")

mm = QuanConv(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
q_output = mm(input)
print(q_output)
'''