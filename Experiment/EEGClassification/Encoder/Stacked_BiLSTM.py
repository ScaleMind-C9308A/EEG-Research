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
                 input_size=128,
                 lstm_layers=2,
                 lstm_size=128,
                 embedding_size=128):
        super().__init__()
        self.lstm_layers = lstm_layers
        self.lstm_size = lstm_size
        self.stacked_bilstm = nn.LSTM(
            input_size, lstm_size, num_layers = self.lstm_layers, bidirectional=True, batch_first = True)
        self.output1 = nn.Linear(2*lstm_size, embedding_size)
        self.relu = nn.ReLU()


    def forward(self, x):
        # Change order of axis from (n, spatial, temporal) to (n, temporal, spatial)
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
        return x