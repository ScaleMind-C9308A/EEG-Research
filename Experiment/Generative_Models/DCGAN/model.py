import torch
import torch.nn as nn

# Define the generator network
class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, 4 * 4 * 512),
            nn.BatchNorm1d(4 * 4 * 512),
            nn.ReLU(True),
            nn.Unflatten(1, (512, 4, 4)),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, input_noise, input_eeg):
        input_combined = torch.cat((input_noise, input_eeg), dim=1)
        return self.main(input_combined)

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, condition_dim):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.fc = nn.Sequential(
            nn.Linear((512+condition_dim)*4*4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1)
            # nn.Sigmoid() # Don't need, since nn.BCELoss() assumes that the input tensor contains logits (raw scores) rather than probabilities
        )

    def forward(self, input_image, input_condition):
        """
        Explain: The condition vector is "spatially appended" after the final conv layer
            => the condition vector needs to be expanded to match the spatial dimensions of the feature map. 
            This can be done using the torch.repeat() or torch.tile() functions.
        """
        x = self.main(input_image)
        batch_size = x.size(0)
        condition = input_condition.view(batch_size, -1, 1, 1)  # reshape condition vector to (batch_size, condition_size, 1, 1)
        condition = condition.repeat(1, 1, x.size(2), x.size(3))  # repeat condition along spatial dimensions to match feature map size
        x = torch.cat((x, condition), dim=1)  # concatenate feature map and condition along the channel dimension
        x = x.view(batch_size, -1)
        output = self.fc(x)
        return output
    
def load_model(noise_dim, condition_dim, is_pretrained_stage1, pretrained_netG, pretrained_netD):
    netG = Generator(noise_dim, condition_dim)
    netD = Discriminator(condition_dim)
    if is_pretrained_stage1:
        netG.load_state_dict(torch.load(pretrained_netG))
        netD.load_state_dict(torch.load(pretrained_netD))
    return netG, netD


