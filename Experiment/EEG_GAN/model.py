import torch.nn as nn
import torch.nn.functional as F
import torch
from util import weight_filler

from modules import conv_cub, conv_lin

def load_model(model_name, weight_path, args):
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

    generator.train_init(alpha=args.lr,betas=(0.,0.99))
    discriminator.train_init(alpha=args.lr,betas=(0.,0.99),eps_center=0.001,
                            one_sided_penalty=True,distance_weighting=True)
    generator = generator.apply(weight_filler)
    discriminator = discriminator.apply(weight_filler)

    
    fade_alpha = 1.
    generator.model.alpha = fade_alpha
    discriminator.model.alpha = fade_alpha
        
    return generator, discriminator