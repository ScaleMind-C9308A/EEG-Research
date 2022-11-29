import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
from scipy.stats import mode
from svmutil import *
##############################################################
# LSTM classifier
##############################################################
class classifier_LSTM(nn.Module):

    def __init__(self,
                 relup,
                 input_size,
                 lstm_layers,
                 lstm_size,
                 output1_size,
                 output2_size,
                 GPUindex):
        super(classifier_LSTM, self).__init__()
        self.relup = relup
        self.lstm_layers = lstm_layers
        self.lstm_size = lstm_size
        self.GPUindex = GPUindex
        self.lstm = nn.LSTM(
            input_size, lstm_size, num_layers = 1, batch_first = True)
        self.output1 = nn.Linear(lstm_size, output1_size)
        self.relu = nn.ReLU()
        if output2_size is None:
            self.output2 = None
        else:
            self.output2 = nn.Linear(lstm_size, output2_size)

    def forward(self, x):
        batch_size = x.size(0)
        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size),
                     torch.zeros(self.lstm_layers, batch_size, self.lstm_size))
        if x.is_cuda:
            lstm_init = (lstm_init[0].cuda(self.GPUindex),
                         lstm_init[0].cuda(self.GPUindex))
        lstm_init = (Variable(lstm_init[0]), Variable(lstm_init[1]))
        x = self.lstm(x, lstm_init)[0][:, -1, :]
        x = self.output1(x)
        if self.relup:
            x = self.relu(x)
        if self.output2 is not None:
            x = self.output2(x)
        return x