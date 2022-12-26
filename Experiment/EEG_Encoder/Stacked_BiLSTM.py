import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
from scipy.stats import mode
##############################################################
# Stacked BiLSTM classifier
##############################################################
class classifier_Stacked_BiLSTM(nn.Module):

    def __init__(self,
                 relup,
                 input_size,
                 lstm_layers,
                 lstm_size,
                 output1_size,
                 output2_size,
                 GPUindex):
        super().__init__()
        self.relup = relup
        self.lstm_layers = lstm_layers
        self.lstm_size = lstm_size
        self.GPUindex = GPUindex
        self.stacked_bilstm = nn.LSTM(
            input_size, lstm_size, num_layers = self.lstm_layers, bidirectional=True, batch_first = True)
        self.output1 = nn.Linear(2*lstm_size, output1_size)
        self.relu = nn.ReLU()
        if output2_size is None:
            self.output2 = None
        else:
            self.output2 = nn.Linear(lstm_size, output2_size)

    def forward(self, x):
        # Change order of axis from (n, 96, 512) to (n, 512, 96)
        x = torch.permute(x, (0, 2, 1))
        batch_size = x.size(0)
        # h0, c0 size are: (D*num_layers, batch, lstm_size); with D=2 for bidirectional RNN
        lstm_init = (torch.zeros(2*self.lstm_layers, batch_size, self.lstm_size),
                     torch.zeros(2*self.lstm_layers, batch_size, self.lstm_size))
        if x.is_cuda:
            lstm_init = (lstm_init[0].cuda(self.GPUindex),
                         lstm_init[0].cuda(self.GPUindex))
        x = self.stacked_bilstm(x, lstm_init)[0] 
        #Output size: (N, Sequence, D*output_size); with D=2 for bidirectional RNN
        lstm_init = (Variable(lstm_init[0]), Variable(lstm_init[1]))
        x = x[:, -1, :] #(N, 1, 2*output_size)
        x = self.output1(x)
        if self.relup:
            x = self.relu(x)
        if self.output2 is not None:
            x = self.output2(x)
        return x