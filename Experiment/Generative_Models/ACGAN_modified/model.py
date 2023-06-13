import torch
import torch.nn as nn
from utils import weights_init
import torch.nn.functional as F


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        if self.bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()
            self.embed.weight.data[:, num_features:].zero_()
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.bn(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=1)
            out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * out
        return out
# Define the generator network
class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim):
        super(GenBlock, self).__init__()
        self.bn1 = ConditionalBatchNorm2d(in_channels, condition_dim)
        self.bn2 = ConditionalBatchNorm2d(out_channels, condition_dim)
        self.activation = nn.ReLU(inplace=True)
        self.conv2d0 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2d2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim):
        super(Generator, self).__init__()
        self.linear0 = nn.Linear(noise_dim + condition_dim, 16384)
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                GenBlock(1024, 512, condition_dim),
            ]),
            nn.ModuleList([
                GenBlock(512, 256, condition_dim),
            ]),
            nn.ModuleList([
                GenBlock(256, 128, condition_dim),
            ]),
            nn.ModuleList([
                GenBlock(128, 64, condition_dim),
            ])
        ])
        self.bn4 = nn.BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = nn.ReLU(inplace=True)
        self.conv2d5 = nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.tanh = nn.Tanh()

    def forward(self, input_noise, input_eeg):
        input_combined = torch.cat((input_noise, input_eeg), dim=1)
        fc1 = self.linear0(input_combined)
        fc1 = fc1.view(input_combined.size(0), -1, 4, 4)
        
        for block in self.blocks:
            for gen_block in block:
                bn1 = gen_block.bn1(input_eeg)
                bn2 = gen_block.bn2(fc1)
                activation = gen_block.activation(bn1)
                conv2d0 = gen_block.conv2d0(activation)
                conv2d1 = gen_block.conv2d1(bn2)
                conv2d2 = gen_block.conv2d2(conv2d1)
                fc1 = conv2d0 + conv2d2
        
        bn4 = self.bn4(fc1)
        activation = self.activation(bn4)
        conv2d5 = self.conv2d5(activation)
        output = self.tanh(conv2d5)
        
        return output

# Define the discriminator network
class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscOptBlock, self).__init__()
        self.conv2d0 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.conv2d1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2d2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.average_pooling = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv2d0(x)
        out = self.conv2d1(x)
        out = self.activation(out)
        out = self.conv2d2(out)
        out = self.bn0(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.average_pooling(out)
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv2d0 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.conv2d1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2d2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.average_pooling = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.activation(x)
        out = self.conv2d0(out)
        out = self.conv2d1(out)
        out = self.activation(out)
        out = self.conv2d2(out)
        out = self.bn0(x)
        out = self.bn1(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.average_pooling(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, num_classes=40):
        super(Discriminator, self).__init__()

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                DiscOptBlock(3, 64),
            ]),
            nn.ModuleList([
                DiscBlock(64, 128),
            ]),
            nn.ModuleList([
                DiscBlock(128, 256),
            ]),
            nn.ModuleList([
                DiscBlock(256, 512),
            ]),
            nn.ModuleList([
                DiscBlock(512, 1024),
            ]),
        ])

        self.activation = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(1024, 1)
        self.linear4 = nn.Linear(1024, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_image):
        x = input_image
        for block in self.blocks:
            for layer in block:
                x = layer(x)
        x = self.activation(x)
        x = x.view(x.size(0), -1)
        realfake = self.sigmoid(self.linear1(x))
        classes = self.softmax(self.linear4(x))
        return realfake, classes
def load_model(num_classes, noise_dim, condition_dim, is_pretrained_stage1, pretrained_netG, pretrained_netD):
    # netG = Generator(noise_dim, condition_dim)
    netG = Generator(noise_dim, condition_dim=num_classes) # for training without EEG condition vector
    netD = Discriminator(num_classes)
    if is_pretrained_stage1:
        netG.load_state_dict(torch.load(pretrained_netG))
        netD.load_state_dict(torch.load(pretrained_netD))
    # else:
        # netG.apply(weights_init)
        # netD.apply(weights_init)
    return netG, netD


