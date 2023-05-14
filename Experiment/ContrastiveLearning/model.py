import torch.nn as nn
import torch.nn.functional as F

from Encoder.eeg_encoder import load_eeg_encoder
from Encoder.image_encoder import load_image_encoder
# from Encoder.image_encoder import

class TripleNet(nn.Module):
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
        img_embed = self.img_encoder(img).logits #for inception_v3
        return eeg_embed, img_embed
    def get_eeg_embedding(self, eeg):
        return self.eeg_encoder(eeg)
    def get_img_embedding(self, img):
        return self.img_encoder(img)
def load_model(model="triple_net", eeg_encoder="EEGChannelNet", img_encoder="inception_v3"):
    """
    model: "triple_net" | "embedding_net"
    """
    eeg_encoder = load_eeg_encoder(eeg_encoder)
    img_encoder = load_image_encoder(img_encoder, 40, pretrained=True)
    print("Image Encoder:")
    print(img_encoder)
    if (model=="triplet_net"):
        model = TripleNet(eeg_encoder, img_encoder)
    elif (model == "embedding_net"):
        model = EmbeddingNet(eeg_encoder, img_encoder)
    return model