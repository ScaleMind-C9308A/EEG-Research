import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
from scipy.stats import mode
##############################################################
# LSTM classifier
##############################################################
class classifier_LSTM_Dropout(nn.Module):

    def __init__(self,
                 input_size=128,
                 lstm_layers=1,
                 lstm_size_1=500,
                 lstm_size_2 = 1000,
                 embedding_size= 128,
                 device=None):
        super(classifier_LSTM_Dropout, self).__init__()
        self.lstm_layers = lstm_layers
        self.lstm_size_1 = lstm_size_1
        self.lstm_size_2 = lstm_size_2
        self.lstm1 = nn.LSTM(
            input_size, lstm_size_1, num_layers = 1, batch_first = True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(lstm_size_1, lstm_size_2, num_layers = 1, batch_first =True)
        self.dropout2 = nn.Dropout(0.2)
        self.output1 = nn.Linear(lstm_size_2, embedding_size)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()
        self.device = device

    def forward(self, x):
        # Change order of axis from (n, 96, 512) to (n, 512, 96)
        x = torch.permute(x, (0, 2, 1))
        batch_size = x.size(0)
        lstm_init1 = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size_1).to(self.device),
                     torch.zeros(self.lstm_layers, batch_size, self.lstm_size_1).to(self.device))
        lstm_init2 = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size_2).to(self.device),
                     torch.zeros(self.lstm_layers, batch_size, self.lstm_size_2).to(self.device))
        # lstm_init = (Variable(lstm_init[0]), Variable(lstm_init[1]))
        x = self.lstm1(x, lstm_init1)[0][:, -1, :]
        x = self.dropout1(x)
        x = self.lstm2(x, lstm_init2)[0][:, -1, :]
        x = self.dropout2(x)
        x = self.output1(x)
        return x
