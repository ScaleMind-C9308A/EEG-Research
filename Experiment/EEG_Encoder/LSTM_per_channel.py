import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchinfo import summary

class classifier_LSTM_per_channel(nn.Module):
    def __init__(self, 
                 channels_num,
                 lstm_layers,
                 lstm_full1_size,
                 lstm_full2_size,
                 output_size,
                 GPUindex):
        super(classifier_LSTM_per_channel, self).__init__()
        self.lstm_size0 = 1
        self.channels_num = channels_num
        self.lstm_layers = lstm_layers
        self.output_size = output_size
        self.GPUindex = GPUindex

        self.lstm_per_channel = nn.LSTM(1, hidden_size=self.lstm_size0, batch_first=True)
        self.lstm_full1 = nn.LSTM(self.channels_num, hidden_size=lstm_full1_size, batch_first=True)
        self.lstm_full2 = nn.LSTM(lstm_full1_size, hidden_size=lstm_full2_size, batch_first=True)
        self.linear = nn.Linear(lstm_full2_size, out_features=self.output_size)
        self.softmax = nn.Softmax()
    def forward(self, X):
        #X: (batch_size, channels_num, sequence_len)
        # print(X.size())
        batch_size = len(X)
        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size0),
                     torch.zeros(self.lstm_layers, batch_size, self.lstm_size0))
        if X.is_cuda:
            lstm_init = (lstm_init[0].cuda(self.GPUindex),
                         lstm_init[0].cuda(self.GPUindex))
        lstm_init = (Variable(lstm_init[0]), Variable(lstm_init[1]))
        first_layer_inputs = []
        second_layer_inputs = [] #(96, timesteps)
        for i in range(self.channels_num): 
            x_in = torch.tensor(X[:, i, :]) # x_in: (batch_size, 1, timesteps)
            x_in = torch.squeeze(x_in, 1) #x_in: (batch_size, timesteps)
            x_in = torch.unsqueeze(x_in, -1) #x_in: (batch_size, timesteps, input_size=1)
            first_layer_inputs.append(x_in)
            x_out = self.lstm_per_channel(x_in, lstm_init)[0] # x_out: (batch_size, timesteps, 1)
            second_layer_inputs.append(x_out)

        X = torch.cat(second_layer_inputs, -1) #X: (batch_size, timesteps, 96)
        # print("X after concat: ", X.size())
        X = self.lstm_full1(X)[0] # output: [batch, 512, hidden_size], (hx, cx)
        X = self.lstm_full2(X)[0][:,-1,:]
        X = self.linear(X)
        X = self.softmax(X)

        return X