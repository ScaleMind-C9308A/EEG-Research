import torch.nn as nn
import torch.nn.functional as F
import torch

from modules.conv_cub import Generator, Discriminator
from modules.conv_lin import Generator, Discriminator

def load_model(model_name, weight_path, num_classes=40, device=None):
    """
    model_name: "conv_cub" | "conv_lin"
    """
    # if (model_name == "conv_cub"):
        
    return model