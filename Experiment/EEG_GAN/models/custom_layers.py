import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# deconvolutional kernel with linear interpolation for multichannel EEG
def linear_kernel(stride=2, in_channels=64, out_channels=128): # stride = 2; num_channels = 1
        filter_size = (2 * stride - stride % 2)
        # Create linear weights in numpy array
        # num_channels = 1
        linear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
        scale_factor = (filter_size + 1) // 2
        if filter_size % 2 == 1:
            center = scale_factor - 1
        else:
            center = scale_factor - 0.5
        for x in range(filter_size):
            for y in range(filter_size):
                linear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * \
                                        (1 - abs(y - center) / scale_factor)
        # weights = np.zeros((filter_size, filter_size, num_channels, num_channels))
        # For pytorch, weights of deconv layer is (in_channels, out_channels/groups, kernel_size[0], kernel_size[1])
        weights = np.zeros((in_channels, out_channels, filter_size, filter_size))
        for i in range(in_channels):
            for j in range(out_channels):
                weights[i, j, :, :] = linear_kernel
        weights = torch.tensor(weights, dtype=torch.float32)
        return weights
class Deconv2D_Linear_Weight(torch.nn.Module):   
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super(Deconv2D_Linear_Weight, self).__init__()
        # Define the convolution operation
        self.layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.stride = stride
        # Define the weight initializer
        nn.init.constant_(self.layer.weight, linear_kernel(stride, in_channels, out_channels))
    
    def forward(self, x):
        # Apply the convolution operation
        x = self.layer(x)

        # Return the output
        return x
    
class ClipLayer(torch.nn.Module):
    def __init__(self, channels, time_step):
        super(ClipLayer, self).__init__()

        self.channels = channels
        self.time_step = time_step

    def forward(self, x):
        # Resize the image with cropping or padding
        x = torch.nn.functional.interpolate(x, (self.channels, self.time_step), mode='bilinear', align_corners=True)

        # Return the output
        return x

class MeanZeroLayer(torch.nn.Module):
    def forward(self, x):
        # Calculate the mean of the image
        mu = x.mean(dim=(2, 0), keepdim=True)

        # Subtract the mean from the image
        x = x - mu

        # Return the output
        return x
    
def gaussian_layer(ins, is_training, mean, stddev):
    if is_training:
        noise = ins.data.new(ins.size()).normal_(mean, stddev)
        return ins + noise
    return ins