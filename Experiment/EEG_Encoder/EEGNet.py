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
        first_kernel = temporal//2 #256
        first_padding = first_kernel//2 #128
        self.network = nn.Sequential( #input(batch, 1, 96, 512)
            ### FIRST BLOCK
            nn.ZeroPad2d((first_padding, first_padding-1, 0, 0)), # (1, 1, 96, 767)
            nn.Conv2d(in_channels = 1,
                      out_channels = F1,
                      kernel_size = (1, first_kernel)), #(1, 8, 96, 512)
            nn.BatchNorm2d(F1),
            ### Depthwise Convolution
            nn.Conv2d(in_channels = F1,
                      out_channels = D*F1,
                      kernel_size = (spatial, 1),
                      groups = F1), #(1, 8, 1, 512)
            ##########################
            # nn.Conv2d(in_channels = F1,
            #           out_channels = D*F1,
            #           kernel_size = 1), #(1, 16, 417, 96)
            nn.BatchNorm2d(D*F1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size = (1, 4)), #(1, 16, 1, 128)
            nn.Dropout(),
            ### SECOND BLOCK
            nn.ZeroPad2d((8, 7, 0, 0)), #(1, 16, 1, 143)
            ### Separable Convolution
            nn.Conv2d(in_channels = D*F1,
                      out_channels = D*F1,
                      kernel_size = (1, 16),
                      groups = F1), #(1, 16, 1, 128)
            nn.Conv2d(in_channels = D*F1,
                      out_channels = F2,
                      kernel_size = 1), #(1, 16, 1, 128)
            #################################
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size = (1, 8)), #(1, 16, 1, 16)
            nn.Dropout())
        self.fc = nn.Linear(F2*(temporal//32), 40) #in=256, out=40

    def forward(self, x):
        # print("First: ", x.size())
        x = x.unsqueeze(0).permute(1, 0, 2, 3)
        # print("After permute: ", x.size())
        x = self.network(x)
        # print("After network: ", x.size())
        x = x.view(x.size()[0], -1)
        # print("After view: ", x.size())
        return self.fc(x)

    def cuda(self, gpuIndex):
        self.network = self.network.cuda(gpuIndex)
        self.fc = self.fc.cuda(gpuIndex)
        return self