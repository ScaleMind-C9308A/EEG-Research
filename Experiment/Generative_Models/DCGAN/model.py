import torch
import torch.nn as nn

# Define the generator network
class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # nn.Linear(noise_dim + condition_dim, 4 * 4 * 512),
            # nn.BatchNorm1d(4 * 4 * 512),
            # nn.ReLU(True),
            # nn.Unflatten(1, (512, 4, 4)),
            
            nn.ConvTranspose2d(noise_dim + condition_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input_noise, input_eeg):
        input_combined = torch.cat((input_noise, input_eeg), dim=1)
        input_combined = input_combined.view(input_combined.size(0), -1, 1, 1)
        output = self.main(input_combined)
        # print(f"Size of generator output: {output.size()}")
        return output

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, condition_dim):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``

        )

        self.fc = nn.Sequential(
            nn.Linear((512+condition_dim)*4*4, 1024),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid() 
        )

    def forward(self, input_image, input_condition):
        """
        Explain: The condition vector is "spatially appended" after the final conv layer
            => the condition vector needs to be expanded to match the spatial dimensions of the feature map. 
            This can be done using the torch.repeat() or torch.tile() functions.
        """
        x = self.main(input_image) #(batch, 512, 4, 4)
        batch_size = x.size(0)
        # transform input_condition to size (batch, 128, 4, 4)
        condition = input_condition.view(batch_size, -1, 1, 1)  # reshape condition vector to (batch_size, condition_size, 1, 1)
        condition = condition.repeat(1, 1, x.size(2), x.size(3))  # repeat condition along spatial dimensions to match feature map size
        # print(f"Transform input condition to size: {condition.size()}")
        # Size after concat is (batch, 512+128, 4, 4)
        x = torch.cat((x, condition), dim=1)  # concatenate feature map and condition along the channel dimension
        # print(f"Size after concat: {x.size()}")
        # Size after flatten is (batch, (512+128)*4*4)
        x = x.view(batch_size, -1)
        # print(f"Size after flatten: {x.size()}")
        output = self.fc(x)
        return output
    
# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def load_model(noise_dim, condition_dim, is_pretrained_stage1, pretrained_netG, pretrained_netD):
    netG = Generator(noise_dim, condition_dim)
    netD = Discriminator(condition_dim)
    if is_pretrained_stage1:
        netG.load_state_dict(torch.load(pretrained_netG))
        netD.load_state_dict(torch.load(pretrained_netD))
    else:
        netG.apply(weights_init)
        netD.apply(weights_init)
    return netG, netD


