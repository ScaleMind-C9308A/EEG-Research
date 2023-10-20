import torch.nn as nn
import torch.nn.functional as F
import torch


from models import WGAN_GP_multichannel

### Class for training:
# GAN_Module => GAN_Discriminator, GAN_Generator => WGAN_*
### Class for Model blocks:
#  Progressive(Dis/Gen)Block + Progressive(Dis/Gen) 
### Define Model 
# conv_lin/conv_cub

def load_model(model_name, weight_path=None, args=None):
    """
    model_name: "wgan_1_channel" | "wgan_multi_channel" | "cc_wgan"
    """
    if model_name == "wgan_1_channel":
        generator, discriminator = WGAN_GP_multichannel.GAN_Generator(), WGAN_GP_multichannel.GAN_Discriminator()
    elif model_name == "wgan_multi_channel":
        generator, discriminator = WGAN_GP_multichannel.GAN_Generator(), WGAN_GP_multichannel.GAN_Discriminator()
    elif model_name == "cc_wgan":
        generator, discriminator = WGAN_GP_multichannel.GAN_Generator(), WGAN_GP_multichannel.GAN_Discriminator()
    else:
        raise ValueError("model_name must be 'wgan_1_channel' | 'wgan_multi_channel' | 'cc_wgan'")        
    return generator, discriminator