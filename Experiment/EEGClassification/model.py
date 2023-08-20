import torch.nn as nn
import torch.nn.functional as F
import torch

from Encoder.eeg_encoder import load_eeg_encoder
from Encoder.image_encoder import load_image_encoder_triplet

class EEGClassificationNet(nn.Module):
    def __init__(self, backbone_name, embedding_dim, num_classes, device):
        super().__init__()
        self.backbone = load_eeg_encoder(backbone_name, embedding_dim, device=device)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, num_classes)
            # # For nn.CrossEntropyLoss() => The input is expected to contain the unnormalized logits for each class
            # nn.LogSoftmax(dim=1)
        )
    def forward(self, eeg):
        return self.backbone(eeg)
class Triplet_EEGClassificationNet(nn.Module):
    def __init__(self, pretrained_model, embedding_dim, num_classes):
        super().__init__()
        for param in pretrained_model.parameters():
            param.requires_grad = False
        self.backbone = pretrained_model
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, num_classes)
            # # For nn.CrossEntropyLoss() => The input is expected to contain the unnormalized logits for each class
            # nn.LogSoftmax(dim=1)
        )
    def forward(self, eeg):
        output = self.backbone.get_eeg_embedding(eeg)            
        output = self.classifier(output)
        return output

class TripletNet(nn.Module):
    def __init__(self, eeg_encoder, img_encoder):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.img_encoder = img_encoder
    def forward(self, eeg, img1, img2):
        eeg_embed = self.eeg_encoder(eeg)
        img1_embed = self.img_encoder(img1)
        img2_embed = self.img_encoder(img2)
        return eeg_embed, img1_embed, img2_embed
    def get_eeg_embedding(self, eeg):
        return self.eeg_encoder(eeg)
    def get_img_embedding(self, img):
        return self.img_encoder(img)

class EmbeddingNet(nn.Module):
    def __init__(self, eeg_encoder, img_encoder):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.img_encoder = img_encoder   
    def forward(self, eeg, img):
        eeg_embed = self.eeg_encoder(eeg)
        img_embed = self.img_encoder(img)
        return eeg_embed, img_embed
    def get_eeg_embedding(self, eeg):
        return self.eeg_encoder(eeg)
    def get_img_embedding(self, img):
        return self.img_encoder(img)
def load_model(mode, weight_path, num_classes=40, eeg_encoder_name="EEGChannelNet", img_encoder_name="inception_v3", output_dim=1000, img_feature_extract=False, device=None):
    """
    mode: "triplet" | "online_triplet" | "classic"
    """
    if (mode == "classic"):
        model = EEGClassificationNet(eeg_encoder_name,output_dim, num_classes, device=device)
    else:
        eeg_encoder = load_eeg_encoder(eeg_encoder_name,output_dim)
        img_encoder = load_image_encoder_triplet(img_encoder_name, output_dim, pretrained=True)
        if (mode =="triplet"):
            backbone = TripletNet(eeg_encoder, img_encoder)
        elif (mode == "online_triplet"):
            backbone = EmbeddingNet(eeg_encoder, img_encoder)
        pretrained_weights = torch.load(weight_path)
        backbone.load_state_dict(pretrained_weights)
        model = Triplet_EEGClassificationNet(backbone, output_dim, num_classes)
    return model