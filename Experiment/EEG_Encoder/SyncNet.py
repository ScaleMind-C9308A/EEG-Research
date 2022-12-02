import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
from scipy.stats import mode
##############################################################
# SyncNet classifier
##############################################################

class classifier_SyncNet(nn.Module):
    def __init__(self, spatial, temporal):
        super(classifier_SyncNet, self).__init__()
        K = min(10, spatial)
        Nt = min(40, temporal)
        pool_size = Nt
        b = np.random.uniform(low = -0.05, high = 0.05, size = (1, spatial, K))
        omega = np.random.uniform(low = 0, high = 1, size = (1, 1, K))
        zeros = np.zeros(shape = (1, 1, K))
        phi_ini = np.random.normal(
            loc = 0, scale = 0.05, size = (1, spatial-1, K))
        phi = np.concatenate([zeros, phi_ini], axis = 1)
        beta = np.random.uniform(low = 0, high = 0.05, size = (1, 1, K))
        t = np.reshape(range(-Nt//2, Nt//2),[Nt, 1, 1])
        tc = np.single(t)
        W_osc = b*np.cos(tc*omega+phi)
        W_decay = np.exp(-np.power(tc, 2)*beta)
        W = W_osc*W_decay
        W = np.transpose(W, (2, 1, 0))
        bias = np.zeros(shape = [K])
        self.net = nn.Sequential(nn.ConstantPad1d((Nt//2, Nt//2-1), 0),
                                 nn.Conv1d(in_channels = spatial,
                                           out_channels = K,
                                           kernel_size = 1,
                                           stride = 1,
                                           bias = True),
                                nn.MaxPool1d(kernel_size = pool_size,
                                             stride = pool_size),
                                nn.ReLU())
        self.net[1].weight.data = torch.FloatTensor(W)
        self.net[1].bias.data = torch.FloatTensor(bias)
        self.fc = nn.Linear((temporal//pool_size)*K, 40)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.view(x.size()[0],-1)
        x = self.fc(x)
        return x

    def cuda(self, gpuIndex):
        self.net = self.net.cuda(gpuIndex)
        self.fc = self.fc.cuda(gpuIndex)
        return self