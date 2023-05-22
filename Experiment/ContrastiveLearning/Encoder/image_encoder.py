from torchvision import models
import torch.nn as nn

def load_image_encoder(backbone, output_dim, pretrained):
    """
    Load image encoder as CNN models

    Args:
        backbone: "inception_v3" | "resnet50" | "efficientnet_b0" |
        output_dim: (default: 40) 
        pretrained: True | False
    """
    return ImageEncoder(backbone, output_dim, pretrained)
    
        
class ImageEncoder(nn.Module):
    def __init__(self, backbone="inception_v3", output_dim=1000, pretrained=True):
        """
        backbone: "inception_v3" | "resnet50" | "efficientnet_b0" |
        output_dim: (default: 40) 
        pretrained: True | False
        """
        super().__init__()
        if pretrained:
            if (backbone == "resnet50"):
                self.backbone = models.resnet50(models.ResNet50_Weights.DEFAULT)
            elif (backbone == "inception_v3"):
                self.backbone = models.inception_v3(models.Inception_V3_Weights.DEFAULT)
            elif (backbone == "efficientnet_b0"):
                self.backbone = models.efficientnet_b0(models.EfficientNet_B0_Weights.DEFAULT)
        else:
            if (backbone == "resnet50"):
                self.backbone = models.resnet50()
            elif (backbone == "inception_v3"):
                self.backbone = models.inception_v3()
            elif (backbone == "efficientnet_b0"):
                self.backbone = models.efficientnet_b0()
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, output_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.backbone(x)