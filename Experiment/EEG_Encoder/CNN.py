import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
from scipy.stats import mode
##############################################################
# CNN classifier
##############################################################
class classifier_CNN(nn.Module):

    def __init__(self, in_channel, timesteps, output_size):
        super(classifier_CNN, self).__init__()
        self.channel = in_channel
        conv1_size = 32
        conv1_stride = 1
        self.conv1_out_channels = 8
        self.conv1_out = int(
            math.floor(((timesteps-conv1_size)/conv1_stride+1)))
        fc1_in = self.channel*self.conv1_out_channels # 96*8
        fc1_out = 40
        pool1_size = 128
        pool1_stride = 64
        pool1_out = int(
            math.floor(((self.conv1_out-pool1_size)/pool1_stride+1)))
        dropout_p = 0.5
        fc2_in = pool1_out*fc1_out
        #input size: (batch*96, 1, 512)
        self.conv1 = nn.Conv1d(in_channels = 1,
                               out_channels = self.conv1_out_channels,
                               kernel_size = conv1_size,
                               stride = conv1_stride) #(batch*96, 8, 481)
        self.fc1 = nn.Linear(fc1_in, fc1_out)
        self.pool1 = nn.AvgPool1d(kernel_size = pool1_size,
                                  stride = pool1_stride)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p = dropout_p)
        self.fc2 = nn.Linear(fc2_in, output_size)

    def forward(self, x):
        batch_size = x.data.shape[0]
        print("Input size: ", x.size())
        #Input size x: (batch, 96, 512)
        x = torch.unsqueeze(x, 2) #(batch, 96, 1, 512)
        print("After unsquueze: ", x.size())
        x = x.contiguous().view(-1, 1, x.data.shape[-1]) #(batch*96,1, 512)
        print("After view: ", x.size())
        x = self.conv1(x) #(batch*96, 8, 481)
        print("After conv1: ", x.size())
        x = self.activation(x)
        x = x.view(batch_size,
                   self.channel,
                   self.conv1_out_channels,
                   self.conv1_out) #(batch, 96, 8, 481)
        print("After view 2: ", x.size())
        x = x.permute(0, 3, 1, 2) #(batch, 481, 96, 8)
        print("After permute 2: ", x.size())
        x = x.contiguous().view(batch_size,
                                self.conv1_out,
                                self.channel*self.conv1_out_channels) #(batch, 481, 96*8)
        print("After view 3: ", x.size())
        x = self.dropout(x)
        x = self.fc1(x) #(batch, 481, 40)
        print("After fc1: ", x.size())
        x = self.dropout(x)
        x = x.permute(0, 2, 1) #(batch, 40, 481)
        print("After permute 3: ", x.size())
        x = self.pool1(x)
        print("After pool1: ", x.size())
        x = x.contiguous().view(batch_size, -1)
        print("After view 4: ", x.size())
        x = self.fc2(x)
        print("After fc2: ", x.size())
        return x