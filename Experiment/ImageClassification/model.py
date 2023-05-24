import torch.nn as nn
import torch.nn.functional as F
import torch

from Encoder.eeg_encoder import load_eeg_encoder
from Encoder.image_encoder import load_image_encoder, load_image_encoder_triplet

class ImageClassificationNet(nn.Module):
    def __init__(self, backbone_name, num_classes=40, feature_extract=True, use_pretrained=True):
        """
        Args:
            backbone_name: "resnet50" | "inception_v3" | ...
        """
        super().__init__()
        self.is_inception = (backbone_name=="inception_v3")
        self.backbone = load_image_encoder(backbone_name, num_classes, feature_extract, use_pretrained)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, img):
        output = self.backbone(img)
        if (self.is_inception and self.training): #check if the model is in training mode
            output = output.logits
        output = self.softmax(output)
        return output
class Triplet_ImageClassificationNet(nn.Module):
    def __init__(self, pretrained_model, embedding_dim, num_classes, is_inception):
        super().__init__()
        self.is_inception = is_inception
        for param in pretrained_model.parameters():
            param.requires_grad = False
        self.backbone = pretrained_model
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, num_classes),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, img):
        output = self.backbone.get_img_embedding(img)
        if (self.is_inception and self.training): #check if the model is in training mode
            output = output.logits
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
def load_model(mode, weight_path, num_classes=40, eeg_encoder_name="EEGChannelNet", img_encoder_name="inception_v3"):
    """
    mode: "triplet" | "online_triplet" | "classic"
    """
    if (mode == "classic"):
        model = ImageClassificationNet(img_encoder_name, num_classes)
    else:
        eeg_encoder = load_eeg_encoder(eeg_encoder_name)
        img_encoder = load_image_encoder_triplet(img_encoder_name, 1000, True)
        if (mode =="triplet"):
            backbone = TripletNet(eeg_encoder, img_encoder)
        elif (mode == "online_triplet"):
            backbone = EmbeddingNet(eeg_encoder, img_encoder)
        pretrained_weights = torch.load(weight_path)
        backbone.load_state_dict(pretrained_weights)
        is_inception = (img_encoder_name=="inception_v3")
        model = Triplet_ImageClassificationNet(backbone, 1000, num_classes, is_inception)
    return model