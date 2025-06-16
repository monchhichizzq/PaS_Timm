import numpy as np
import torch
import torch.nn as nn
from core_sparse.layers.spa_layer import Input_Placeholder




class SparseConv2d(Input_Placeholder, nn.Conv2d):
    """
    Sparse convolutional layer that applies a threshold to the input tensor.
    """
    def __init__(self, conv2d_layer, threshold, inplace):
        Input_Placeholder.__init__(self, thresh=threshold, inplace=inplace)
        nn.Conv2d.__init__(self, conv2d_layer.in_channels, conv2d_layer.out_channels,
                           kernel_size=conv2d_layer.kernel_size, stride=conv2d_layer.stride,
                           padding=conv2d_layer.padding, dilation=conv2d_layer.dilation,
                           groups=conv2d_layer.groups, bias=conv2d_layer.bias is not None)

        if conv2d_layer.weight is not None:
            self.weight.data = conv2d_layer.weight.data

        if conv2d_layer.bias is not None:
            self.bias.data = conv2d_layer.bias.data
        
    def forward(self, x):
        # Apply thresholding to the input tensor
        x = Input_Placeholder.forward(self, x)

        # Perform convolution
        x = nn.Conv2d.forward(self, x)
        
        return x


class SparseConvTranspose2d(Input_Placeholder, nn.ConvTranspose2d):
    """
    Sparse convolutional layer that applies a threshold to the input tensor.
    """
    def __init__(self, conv2d_layer, threshold, inplace):
        Input_Placeholder.__init__(self, thresh=threshold, inplace=inplace)
        nn.Conv2d.__init__(self, conv2d_layer.in_channels, conv2d_layer.out_channels,
                           kernel_size=conv2d_layer.kernel_size, stride=conv2d_layer.stride,
                           padding=conv2d_layer.padding, dilation=conv2d_layer.dilation,
                           groups=conv2d_layer.groups, bias=conv2d_layer.bias is not None)

        if conv2d_layer.weight is not None:
            self.weight.data = conv2d_layer.weight.data

        if conv2d_layer.bias is not None:
            self.bias.data = conv2d_layer.bias.data
        
    def forward(self, x):
        # Apply thresholding to the input tensor
        x = Input_Placeholder.forward(self, x)
        
        # Perform convolution
        x = nn.Conv2d.forward(self, x)
        
        return x


class SparseLinear(Input_Placeholder, nn.Linear):
    """
    Sparse linear layer that applies a threshold to the input tensor.
    """
    def __init__(self, linear_layer, threshold, inplace):
        Input_Placeholder.__init__(self, thresh=threshold, inplace=inplace)
        nn.Linear.__init__(self, linear_layer.in_features, linear_layer.out_features, bias=linear_layer.bias is not None)

        if linear_layer.weight is not None:
            self.weight.data = linear_layer.weight.data

        if linear_layer.bias is not None:
            self.bias.data = linear_layer.bias.data
        
    def forward(self, x):
        # Apply thresholding to the input tensor
        x = Input_Placeholder.forward(self, x)
        
        # Perform linear transformation
        x = nn.Linear.forward(self, x)
        
        return x
    
class SparseBatchNorm2d(Input_Placeholder, nn.BatchNorm2d):
    """
    Sparse batch normalization layer that applies a threshold to the input tensor.
    """
    def __init__(self, batchnorm_layer, threshold, inplace):
        Input_Placeholder.__init__(self, thresh=threshold, inplace=inplace)
        nn.BatchNorm2d.__init__(self, num_features=batchnorm_layer.num_features)

        self.weight = nn.parameter(batchnorm_layer.weight.data)
        self.bias = nn.parameter(batchnorm_layer.bias.data)
        self.running_mean = nn.parameter(batchnorm_layer.running_mean.data)
        self.running_var = nn.parameter(batchnorm_layer.running_var.data)
        self.eps = batchnorm_layer.eps
        self.momentum = batchnorm_layer.momentum
        self.affine = batchnorm_layer.affine
        self.track_running_stats = batchnorm_layer.track_running_stats
        self.num_batches_tracked = batchnorm_layer.num_batches_tracked  
        
    def forward(self, x):
        # Apply thresholding to the input tensor
        x = Input_Placeholder.forward(self, x)
        
        # Perform batch normalization
        x = nn.BatchNorm2d.forward(self, x)
        
        return x

