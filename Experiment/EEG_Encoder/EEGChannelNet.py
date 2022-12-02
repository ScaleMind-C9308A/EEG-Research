import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
from scipy.stats import mode
#############################################################################
# EEG-ChannelNet classifier
#############################################################################

class classifier_EEGChannelNet(nn.Module):

    def __init__(self, spatial, temporal):
        super(classifier_EEGChannelNet, self).__init__()
        self.temporal_layers = []
        self.temporal_layers.append(nn.Sequential(nn.Conv2d(in_channels = 1,
                                    out_channels = 10,
                                    kernel_size = (1, 33),
                                    stride = (1, 2),
                                    dilation = (1, 1),
                                    padding = (0, 16)),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU()))
        self.temporal_layers.append(nn.Sequential(nn.Conv2d(in_channels = 1,
                                    out_channels = 10,
                                    kernel_size = (1, 33),
                                    stride = (1, 2),
                                    dilation = (1, 2),
                                    padding = (0, 32)),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU()))
        self.temporal_layers.append(nn.Sequential(nn.Conv1d(in_channels = 1,
                                    out_channels = 10,
                                    kernel_size = (1, 33),
                                    stride = (1, 2),
                                    dilation = (1, 4),
                                    padding = (0, 64)),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU()))
        self.temporal_layers.append(nn.Sequential(nn.Conv1d(in_channels = 1,
                                    out_channels = 10,
                                    kernel_size = (1, 33),
                                    stride = (1, 2),
                                    dilation = (1, 8),
                                    padding = (0, 128)),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU()))
        self.temporal_layers.append(nn.Sequential(nn.Conv1d(in_channels = 1,
                                    out_channels = 10,
                                    kernel_size = (1, 33),
                                    stride = (1, 2),
                                    dilation = (1, 16),
                                    padding = (0, 256)),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU()))
        self.spatial_layers = []
        self.spatial_layers.append(nn.Sequential(nn.Conv2d(in_channels = 50,
                                   out_channels = 50,
                                   kernel_size = (128, 1),
                                   stride = (2, 1),
                                   padding = (63, 0)),
                                   nn.BatchNorm2d(50),
                                   nn.ReLU()))
        self.spatial_layers.append(nn.Sequential(nn.Conv2d(in_channels = 50,
                                   out_channels = 50,
                                   kernel_size = (64, 1),
                                   stride = (2, 1),
                                   padding = (31, 0)),
                                   nn.BatchNorm2d(50),
                                   nn.ReLU()))
        self.spatial_layers.append(nn.Sequential(nn.Conv2d(in_channels = 50,
                                   out_channels = 50,
                                   kernel_size = (32, 1),
                                   stride = (2, 1),
                                   padding = (15, 0)),
                                   nn.BatchNorm2d(50),
                                   nn.ReLU()))
        self.spatial_layers.append(nn.Sequential(nn.Conv2d(in_channels = 50,
                                   out_channels = 50,
                                   kernel_size = (16, 1),
                                   stride = (2, 1),
                                   padding = (7, 0)),
                                   nn.BatchNorm2d(50),
                                   nn.ReLU()))
        self.residual_layers = []
        self.residual_layers.append(nn.Sequential(nn.Conv2d(in_channels = 200,
                                    out_channels = 200,
                                    kernel_size = 3,
                                    stride = 2,
                                    padding = 1),
                                    nn.BatchNorm2d(200),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 200,
                                    out_channels = 200,
                                    kernel_size = 3,
                                    stride = 1,
                                    padding = 1),
                                    nn.BatchNorm2d(200)))
        self.residual_layers.append(nn.Sequential(nn.Conv2d(in_channels = 200,
                                    out_channels = 200,
                                    kernel_size = 3,
                                    stride = 2,
                                    padding = 1),
                                    nn.BatchNorm2d(200),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 200,
                                    out_channels = 200,
                                    kernel_size = 3,
                                    stride = 1,
                                    padding = 1),
                                    nn.BatchNorm2d(200)))
        self.residual_layers.append(nn.Sequential(nn.Conv2d(in_channels = 200,
                                    out_channels = 200,
                                    kernel_size = 3,
                                    stride = 2,
                                    padding = 1),
                                    nn.BatchNorm2d(200),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 200,
                                    out_channels = 200,
                                    kernel_size = 3,
                                    stride = 1,
                                    padding = 1),
                                    nn.BatchNorm2d(200)))
        self.residual_layers.append(nn.Sequential(nn.Conv2d(in_channels = 200,
                                    out_channels = 200,
                                    kernel_size = 3,
                                    stride = 2,
                                    padding = 1),
                                    nn.BatchNorm2d(200),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 200,
                                    out_channels = 200,
                                    kernel_size = 3,
                                    stride = 1,
                                    padding = 1),
                                    nn.BatchNorm2d(200)))
        self.shortcuts = []
        self.shortcuts.append(nn.Sequential(nn.Conv2d(in_channels = 200,
                              out_channels = 200,
                              kernel_size = 1,
                              stride = 2),
                              nn.BatchNorm2d(200)))
        self.shortcuts.append(nn.Sequential(nn.Conv2d(in_channels = 200,
                              out_channels = 200,
                              kernel_size = 1,
                              stride = 2),
                              nn.BatchNorm2d(200)))
        self.shortcuts.append(nn.Sequential(nn.Conv2d(in_channels = 200,
                              out_channels = 200,
                              kernel_size = 1,
                              stride = 2),
                              nn.BatchNorm2d(200)))
        self.shortcuts.append(nn.Sequential(nn.Conv2d(in_channels = 200,
                              out_channels = 200,
                              kernel_size = 1,
                              stride = 2),
                              nn.BatchNorm2d(200)))
        spatial_kernel = 3
        temporal_kernel = 3
        if spatial == 128:
            spatial_kernel = 3
        elif spatial==96:
            spatial_kernel = 3
        elif spatial==64:
            spatial_kernel = 2
        else:
            spatial_kernel = 1
        if temporal == 1024:
            temporal_kernel = 3
        elif temporal == 512:
            temporal_kernel = 3
        elif temporal == 440:
            temporal_kernel = 3
        elif temporal == 50:
            temporal_kernel = 2
        self.final_conv = nn.Conv2d(in_channels = 200,
                                    out_channels = 50,
                                    kernel_size = (spatial_kernel,
                                                   temporal_kernel),
                                    stride = 1,
                                    dilation = 1,
                                    padding = 0)
        spatial_sizes = [128, 96, 64, 32, 16, 8]
        spatial_outs = [2, 1, 1, 1, 1, 1]
        temporal_sizes = [1024, 512, 440, 256, 200, 128, 100, 50]
        temporal_outs = [30, 14, 12, 6, 5, 2, 2, 1]
        inp_size = (50*
                    spatial_outs[spatial_sizes.index(spatial)]*
                    temporal_outs[temporal_sizes.index(temporal)])
        self.fc1 = nn.Linear(inp_size, 1000)
        self.fc2 = nn.Linear(1000, 40)

    def forward(self, x):
        x = x.unsqueeze(0).permute(1, 0, 3, 2)
        y = []
        for i in range(5):
            y.append(self.temporal_layers[i](x))
        x = torch.cat(y, 1)
        y=[]
        for i in range(4):
            y.append(self. spatial_layers[i](x))
        x = torch.cat(y, 1)
        for i in range(4):
            x = F.relu(self.shortcuts[i](x)+self.residual_layers[i](x))
        x = self.final_conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def cuda(self, gpuIndex):
        for i in range(len(self.temporal_layers)):
            self.temporal_layers[i] = self.temporal_layers[i].cuda(gpuIndex)
        for i in range(len(self.spatial_layers)):
            self.spatial_layers[i] = self.spatial_layers[i].cuda(gpuIndex)
        for i in range(len(self.residual_layers)):
            self.residual_layers[i] = self.residual_layers[i].cuda(gpuIndex)
        for i in range(len(self.shortcuts)):
            self.shortcuts[i] = self.shortcuts[i].cuda(gpuIndex)
        self.final_conv = self.final_conv.cuda(gpuIndex)
        self.fc1 = self.fc1.cuda(gpuIndex)
        self.fc2 = self.fc2.cuda(gpuIndex)
        return self