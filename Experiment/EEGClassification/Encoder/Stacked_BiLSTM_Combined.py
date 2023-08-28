import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
##############################################################
# Stacked BiLSTM classifier
##############################################################
class Encoder_Stacked_BiLSTM_Combined(nn.Module):

    def __init__(self,
                 input_size=128,
                 bilstm_layers=50,
                 lstm_layers_1=50,
                 lstm_size=128,
                 embedding_size=128,
                 device=None):
        super().__init__()
        self.bilstm_layers = bilstm_layers
        self.lstm_layers_1 = lstm_layers_1
        self.lstm_size = lstm_size
        self.stacked_bilstm = nn.LSTM(
            input_size, lstm_size, num_layers = self.bilstm_layers, bidirectional=True, batch_first = True)
        self.stacked_lstm_1 = nn.LSTM(
            lstm_size*2, lstm_size, num_layers = self.lstm_layers_1, bidirectional=False, batch_first = True)
        self.output1 = nn.Linear(lstm_size, embedding_size)
        self.relu = nn.ReLU()
        self.device = device

    def forward(self, x):
        # Change order of axis from (n, spatial, temporal) to (n, temporal, spatial)
        x = torch.permute(x, (0, 2, 1))
        batch_size = x.size(0)
        # h0, c0 size are: (D*num_layers, batch, lstm_size); with D=2 for bidirectional RNN
        lstm_init1 = (torch.zeros(2*self.lstm_layers, batch_size, self.lstm_size).to(self.device),
                     torch.zeros(2*self.lstm_layers, batch_size, self.lstm_size).to(self.device))
        lstm_init2 = (torch.zeros(2*self.lstm_layers, batch_size, self.lstm_size_2).to(self.device),
                     torch.zeros(2*self.lstm_layers, batch_size, self.lstm_size_2).to(self.device))
        #Output size: (N, Sequence, D*output_size); with D=2 for bidirectional RNN
        x = self.stacked_bilstm(x, lstm_init1)[0] 
        x = self.stacked_lstm_1(x, lstm_init2)[0]
        x = x[:, -1, :] #(N, 1, 2*output_size)
        x = self.output1(x)
        x = self.relu(x)
        return x