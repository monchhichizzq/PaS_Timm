import numpy as np
import torch
import torch.nn as nn
from core_sparse.layers.spa_layer import Input_Threshold
from core_sparse.layers.prune_staic_layer import BinaryConv2d
from core_sparse.layers.prune_dynamic_layer import BinaryAttention



class PaSAct(BinaryConv2d, Input_Threshold):
    def __init__(self, act_layer, in_chs):
        Input_Threshold.__init__(self, thresh=-25)
        BinaryConv2d.__init__(self, 
                            in_channels=in_chs, 
                            out_channels=in_chs,
                            groups=in_chs,
                            kernel_size=1
                            )
        self.act_layer = act_layer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        # Apply activation function
        x = self.act_layer(x)

        # Apply binary to the input tensor
        x = BinaryConv2d.forward(self, x)

        # Apply thresholding to the input tensor
        x = Input_Threshold.forward(self, x)
        return x


class DPaSAct(BinaryAttention, Input_Threshold):
    def __init__(self, act_layer, in_chs):
        Input_Threshold.__init__(self, thresh=-25)
        BinaryAttention.__init__(self, 
                            in_channels=in_chs, 
                            out_channels=in_chs,
                            )
        self.act_layer = act_layer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        # Apply activation function
        x = self.act_layer(x)

        # Apply binary to the input tensor
        att_mask = BinaryAttention.forward(self, x)
        x = x * att_mask

        # Apply thresholding to the input tensor
        # x = Input_Threshold.forward(self, x)
        return x






# backup
# class PaSAct(nn.Module):
#     def __init__(self, act_layer, in_chs):
#         super(PaSAct, self).__init__()
#         self.binary_conv = BinaryConv2d(in_channels=in_chs,
#             out_channels=in_chs,
#             kernel_size=1,
#             groups=in_chs)
#         self.act_layer = act_layer
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     def forward(self, x):
#         # Apply activation function
#         x = self.act_layer(x)

#         # Apply binary to the input tensor
#         x = self.binary_conv(x)

#         # Apply thresholding to the input tensor
#         # x = Input_Threshold.forward(self, x)
#         return x

# class PaSAct(nn.Module):
#     def __init__(self, act_layer):
#         super(PaSAct, self).__init__()
#         Input_Threshold.__init__(self, thresh=-25)
#         self.act_layer = act_layer
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
#     def forward(self, x):
#         # Apply activation function
#         x = self.act_layer(x)

#         # Apply binary to the input tensor
#         in_chs = x.shape[1]
#         self.binary_conv = BinaryConv2d(
#             in_channels=in_chs,
#             out_channels=in_chs,
#             kernel_size=1,
#             groups=in_chs)
#         self.binary_conv.to(self.device)
#         x = self.binary_conv(x)
#         # print("x.shape")

#         # Apply thresholding to the input tensor
#         x = Input_Threshold.forward(self, x)
#         return x
  