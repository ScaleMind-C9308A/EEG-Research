import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
from scipy.stats import mode
##############################################################
# EEGNet classifier
##############################################################

class classifier_EEGNet(nn.Module):
    def __init__(self, spatial, temporal):
        super(classifier_EEGNet, self).__init__()
        #possible spatial [128, 96, 64, 32, 16, 8]
        #possible temporal [1024, 512, 440, 256, 200, 128, 100, 50]
        F1 = 8
        F2 = 16
        D = 2
        first_kernel = temporal//2
        first_padding = first_kernel//2
        self.network = nn.Sequential(
            nn.ZeroPad2d((first_padding, first_padding-1, 0, 0)),
            nn.Conv2d(in_channels = 1,
                      out_channels = F1,
                      kernel_size = (1, first_kernel)),
            nn.BatchNorm2d(F1),
            nn.Conv2d(in_channels = F1,
                      out_channels = F1,
                      kernel_size = (spatial, 1),
                      groups = F1),
            nn.Conv2d(in_channels = F1,
                      out_channels = D*F1,
                      kernel_size = 1),
            nn.BatchNorm2d(D*F1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size = (1, 4)),
            nn.Dropout(),
            nn.ZeroPad2d((8, 7, 0, 0)),
            nn.Conv2d(in_channels = D*F1,
                      out_channels = D*F1,
                      kernel_size = (1, 16),
                      groups = F1),
            nn.Conv2d(in_channels = D*F1,
                      out_channels = F2,
                      kernel_size = 1),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size = (1, 8)),
            nn.Dropout())
        self.fc = nn.Linear(F2*(temporal//32), 40)

    def forward(self, x):
        x = x.unsqueeze(0).permute(1, 0, 3, 2)
        x = self.network(x)
        x = x.view(x.size()[0], -1)
        return self.fc(x)

    def cuda(self, gpuIndex):
        self.network = self.network.cuda(gpuIndex)
        self.fc = self.fc.cuda(gpuIndex)
        return self