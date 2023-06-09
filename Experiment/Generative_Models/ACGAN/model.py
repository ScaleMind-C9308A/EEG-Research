import torch
import torch.nn as nn
from utils import weights_init
# Define the generator network
class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, 768),
            nn.ReLU(True)
        )
        self.main = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 5, 2, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True),

            nn.ConvTranspose2d(384, 256, 5, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 192, 5, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(192, 64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, 8, 2, 0, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input_noise, input_eeg):
        input_combined = torch.cat((input_noise, input_eeg), dim=1)
        fc1 = self.fc1(input_combined)
        fc1 = fc1.view(input_combined.size(0), -1, 1, 1)
        output = self.main(fc1)
        # print(f"Size of generator output: {output.size()}")
        return output

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, num_classes=40):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),

            nn.Conv2d(16, 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),

            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),

            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),

            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),

            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )

        # self.fc = nn.Sequential(
        #     nn.Linear((512+condition_dim)*4*4, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1),
        #     nn.Sigmoid() 
        # )

        # discriminator fc
        self.fc_dis = nn.Linear(13*13*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(13*13*512, num_classes)
        # softmax and sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_image, input_condition=None):
        """
        Explain: The condition vector is "spatially appended" after the final conv layer
            => the condition vector needs to be expanded to match the spatial dimensions of the feature map. 
            This can be done using the torch.repeat() or torch.tile() functions.
        """
        x = self.main(input_image) #(batch, 512, 4, 4)
        batch_size = x.size(0)

        # ################
        # # SPTIALLY APPEND CONDITION VECTOR TO LAST LAYER OF DISCRIMINATOR
        # ###############
        # # transform input_condition to size (batch, 128, 4, 4)
        # condition = input_condition.view(batch_size, -1, 1, 1)  # reshape condition vector to (batch_size, condition_size, 1, 1)
        # condition = condition.repeat(1, 1, x.size(2), x.size(3))  # repeat condition along spatial dimensions to match feature map size
        # # print(f"Transform input condition to size: {condition.size()}")
        # # Size after concat is (batch, 512+128, 4, 4)
        # x = torch.cat((x, condition), dim=1)  # concatenate feature map and condition along the channel dimension
        # # print(f"Size after concat: {x.size()}")
        # # Size after flatten is (batch, (512+128)*4*4)
        # x = x.view(batch_size, -1)        
        # output = self.fc(x)

        flat = x.view(batch_size, -1)
        fc_dis = self.fc_dis(flat)
        fc_aux = self.fc_aux(flat)
        classes = self.softmax(fc_aux)
        realfake = self.sigmoid(fc_dis)
        return realfake, classes
    
def load_model(num_classes, noise_dim, condition_dim, is_pretrained_stage1, pretrained_netG, pretrained_netD):
    netG = Generator(noise_dim, condition_dim)
    netD = Discriminator(num_classes)
    if is_pretrained_stage1:
        netG.load_state_dict(torch.load(pretrained_netG))
        netD.load_state_dict(torch.load(pretrained_netD))
    else:
        netG.apply(weights_init)
        netD.apply(weights_init)
    return netG, netD


