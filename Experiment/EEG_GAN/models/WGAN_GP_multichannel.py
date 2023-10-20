import torch
import torch.nn as nn
import torch.nn.functional as F
from models.custom_layers import Deconv2D_Linear_Weight, ClipLayer, MeanZeroLayer, gaussian_layer, Conv2dSamePadding

class EEGGenerator(nn.Module):
    def __init__(self, n_channels=64, n_time_steps=64, n_z=120):
        super(EEGGenerator, self).__init__()

        self.fc_layer1 = nn.Sequential(
            nn.Linear(n_z, 1024),
            nn.LeakyReLU(),
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(1024, 128 * 18 * 18),
            nn.BatchNorm1d(128 * 18 * 18),
            nn.LeakyReLU(),
        )
        ###
        #  The first upsampling step uses a bi-cubical interpolation followed by a
        # convolutional layer
        ###
        self.upsample_layer1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        ###
        # The second upsampling step uses a deconvolution with bi-linear weights initialization
        ###
        self.upsample_layer2 = nn.Sequential(
            Deconv2D_Linear_Weight(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2),
            ClipLayer(channels=n_channels, time_step=n_time_steps),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )        
        
        # Define the final layer
        self.conv_layer2 = nn.Sequential(
            Conv2dSamePadding(in_channels=128, out_channels=1, kernel_size=(3, 3)),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = x.view(-1, 128, 18, 18)
        x = self.upsample_layer1(x)
        x = self.conv_layer1(x)
        x = self.upsample_layer2(x)
        x = self.conv_layer2(x)
        return x


class EEGDiscriminator(nn.Module):
    def __init__(self):
        super(EEGDiscriminator, self).__init__()

        ###
        # Conv layers
        ###
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding='same'),
            nn.LeakyReLU(0.2),
        )
        self.conv_layer2 = nn.Sequential(
            Conv2dSamePadding(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2),
            nn.LeakyReLU(0.2),
        )
        self.conv_layer3 = nn.Sequential(
            Conv2dSamePadding(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=2),
            nn.LeakyReLU(0.2),
        )
        ###
        # Fully connected layers
        ###
        self.fc_layer1 = nn.Linear(128 * 16 * 16, 1024)
        self.fc_layer2 = nn.Linear(1024, 1)
        ###
        # Other layers
        ###
        self.flatten = nn.Flatten()
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # firstly reshape the input from (batch_size, 64, 64) to (batch_size, 1, 64, 64)
        x = x.view(-1, 1, 64, 64)
        # add gaussian noise to the input for regularization (only in training mode)
        x = gaussian_layer(x, self.training, 0, 0.05)
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)

        x = self.flatten(x)

        x = self.fc_layer1(x)
        x = self.leaky_relu(x)

        x = self.fc_layer2(x)

        return x
