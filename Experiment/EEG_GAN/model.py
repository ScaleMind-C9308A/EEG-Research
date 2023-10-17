import torch.nn as nn
import torch.nn.functional as F
import torch


from modules import conv_cub, conv_lin

### Class for training:
# GAN_Module => GAN_Discriminator, GAN_Generator => WGAN_*
### Class for Model blocks:
#  Progressive(Dis/Gen)Block + Progressive(Dis/Gen) 
### Define Model 
# conv_lin/conv_cub

def load_model(model_name, weight_path=None, args=None):
    """
    model_name: "conv_cub" | "conv_lin"
    """
    n_z = 200
    n_blocks = 6
    if (model_name == "conv_cub"):
        generator = conv_cub.Generator(128, n_z)
        discriminator = conv_cub.Discriminator(128)
    elif (model_name == "conv_lin"):
        generator = conv_lin.Generator(128, n_z)
        discriminator = conv_lin.Discriminator(128)        
    return generator, discriminator