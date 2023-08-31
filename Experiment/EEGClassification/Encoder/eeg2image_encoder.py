import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np

from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from image_encoder import load_image_encoder

class Encoder_EEG2Image(nn.Sequential):
    def __init__(self, image_encoder, output_dim):
        super().__init__()
        self.image_encoder = load_image_encoder(image_encoder, output_dim, feature_extract=False, pretrained=False)
