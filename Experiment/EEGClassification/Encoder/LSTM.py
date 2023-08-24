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
class classifier_LSTM(nn.Module):

    def __init__(self,
                 input_size=128,
                 lstm_layers=1,
                 lstm_size=128,
                 embedding_size=128,
                 device=None):
        super(classifier_LSTM, self).__init__()
        self.lstm_layers = lstm_layers
        self.lstm_size = lstm_size
        self.lstm = nn.LSTM(
            input_size, lstm_size, num_layers = 1, batch_first = True)
        self.output1 = nn.Linear(lstm_size, embedding_size)
        self.relu = nn.ReLU()
        self.device = device

    def forward(self, x):
        # Change order of axis from (n, 96, 512) to (n, 512, 96)
        x = torch.permute(x, (0, 2, 1))
        batch_size = x.size(0)
        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size).to(self.device),
                     torch.zeros(self.lstm_layers, batch_size, self.lstm_size).to(self.device))
        # lstm_init = (Variable(lstm_init[0]), Variable(lstm_init[1]))
        x = self.lstm(x, lstm_init)[0][:, -1, :]
        x = self.output1(x)
        # EDIT: Add relu
        x = self.relu(x)
        return x
