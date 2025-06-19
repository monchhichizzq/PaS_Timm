import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class Check_G(Function):
    @staticmethod
    def forward(ctx, x):
        # print("x:", torch.min(x), torch.max(x))
        return x
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * 1e2 # search scale
        # print("grad_output:", torch.min(grad_output), torch.max(grad_output))
        return grad_output      

class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(BinaryConv2d, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, 
                dilation, groups, bias)
        nn.init.constant_(self.weight, 1.0) # 0.6
        self.prune_thr = 0.5

    def forward(self, x):
        # get raw weights
        raw_w = self.weight.detach()
        # binarize weights
        bin_w = (raw_w > self.prune_thr).float()
        # get changes between raw and binary weights
        res_w = raw_w - bin_w
        new_w = self.weight - res_w # raw_w - (raw_w - bin_w) = bin_w
        # print("new_w:", torch.min(new_w), torch.max(new_w))
        # print("weights:", torch.min(self.weight), torch.max(self.weight), self.weight.requires_grad)
        new_w = Check_G.apply(new_w)
        # build binary depthwise convolution
        output = F.conv2d(x, new_w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output



# class BinaryConv2d(nn.Conv2d):
#     """docstring for QuanConv"""

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=False):
#         super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
#                                            groups, bias)
#         nn.init.constant_(self.weight, 0.6)

#     # @weak_script_method
#     def forward(self, x):
#         # weight = self.weight
#         w = self.weight.detach()
#         binary_w = (w > 0.5).float()
#         residual = w - binary_w
#         weight = self.weight - residual # self.weight - w + binary_w = binary_w
#         print("weight:", torch.min(weight), torch.max(weight), weight.shape)
#         output = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         return output
