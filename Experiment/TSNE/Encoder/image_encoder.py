from torchvision import models
import torch.nn as nn
from Encoder.efficientnet_b0_lite import get_model

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
def load_image_encoder(model_name, output_dim, feature_extract, use_pretrained=True):
    """
    Initialize these variables which will be set in this if statement. Each of these
    variables is model specific.

    Args:
        model_name: "inception_v3" | "resnet50" | "efficientnet_b0" | "efficientnet_b1" |
        output_dim: (default: 40) 
        use_pretrained: True | False
        feature_extract: True | False
    """
    model_ft = None
    input_size = 0

    if model_name == "resnet50":
        """ Resnet18
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, output_dim)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,output_dim)
        input_size = 224

    elif model_name == "efficientnet_b0":
        """ Efficientnet B0
        """
        model_ft = models.efficientnet_b0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs,output_dim)
        input_size = 224

    elif model_name == "densenet_121":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, output_dim)
        input_size = 224

    elif model_name == "inception_v3":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # # Handle the auxilary net
        # num_ftrs = model_ft.AuxLogits.fc.in_features
        # model_ft.AuxLogits.fc = nn.Linear(num_ftrs, output_dim)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,output_dim)
        input_size = 299
    elif model_name == "efficientnet_b0_lite":
        """ Efficientnet B0_lite
        """
        model_ft = get_model(output_dim)
        set_parameter_requires_grad(model_ft, False)
        # num_ftrs = model_ft.classifier[1].in_features
        # model_ft.classifier[1] = nn.Linear(num_ftrs,output_dim)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

def load_image_encoder_triplet(backbone, output_dim, pretrained):
    """
    Load image encoder as CNN models

    Args:
        backbone: "inception_v3" | "resnet50" | "efficientnet_b0" | "efficientnet_b1" |
        output_dim: (default: 40) 
        pretrained: True | False
        feature_extract: True | False
    """
    return ImageEncoder(backbone, output_dim, pretrained)
    
        
class ImageEncoder(nn.Module):
    def __init__(self, backbone="inception_v3", output_dim=500, pretrained=True):
        """
        backbone: "inception_v3" | "resnet50" | "efficientnet_b0" | "efficientnet_b1" |
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
            elif (backbone == "efficientnet_b1"):
                self.backbone = models.efficientnet_b1(models.EfficientNet_B1_Weights.DEFAULT)
        else:
            if (backbone == "resnet50"):
                self.backbone = models.resnet50()
            elif (backbone == "inception_v3"):
                self.backbone = models.inception_v3()
            elif (backbone == "efficientnet_b0"):
                self.backbone = models.efficientnet_b0()
            elif(backbone == "efficientnet_b1"):
                self.backbone = models.efficientnet_b1()
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, output_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.backbone(x)
    