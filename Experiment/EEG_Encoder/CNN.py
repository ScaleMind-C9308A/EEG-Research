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

    def __init__(self, in_channel, num_points, output_size):
        super(classifier_CNN, self).__init__()
        self.channel = in_channel
        conv1_size = 32
        conv1_stride = 1
        self.conv1_out_channels = 8
        self.conv1_out = int(
            math.floor(((num_points-conv1_size)/conv1_stride+1)))
        fc1_in = self.channel*self.conv1_out_channels
        fc1_out = 40
        pool1_size = 128
        pool1_stride = 64
        pool1_out = int(
            math.floor(((self.conv1_out-pool1_size)/pool1_stride+1)))
        dropout_p = 0.5
        fc2_in = pool1_out*fc1_out
        self.conv1 = nn.Conv1d(in_channels = 1,
                               out_channels = self.conv1_out_channels,
                               kernel_size = conv1_size,
                               stride = conv1_stride)
        self.fc1 = nn.Linear(fc1_in, fc1_out)
        self.pool1 = nn.AvgPool1d(kernel_size = pool1_size,
                                  stride = pool1_stride)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p = dropout_p)
        self.fc2 = nn.Linear(fc2_in, output_size)

    def forward(self, x):
        batch_size = x.data.shape[0]
        x = x.permute(0, 2, 1)
        x = torch.unsqueeze(x, 2)
        x = x.contiguous().view(-1, 1, x.data.shape[-1])
        x = self.conv1(x)
        x = self.activation(x)
        x = x.view(batch_size,
                   self.channel,
                   self.conv1_out_channels,
                   self.conv1_out)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch_size,
                                self.conv1_out,
                                self.channel*self.conv1_out_channels)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.pool1(x)
        x = x.contiguous().view(batch_size, -1)
        x = self.fc2(x)
        return x