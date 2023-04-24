from torchvision import models

def load_image_encoder(model_name, pretrained):
    """
    Load image encoder as CNN models

    Args:
        model_name: resnet50 | inception_v3
        pretrained: True | False
    """
    if pretrained:
        if (model_name == "resnet50"):
            return models.resnet50(models.ResNet50_Weights)
        if (model_name == "inception_v3"):
            return models.inception_v3(models.Inception_V3_Weights)
    else:
        if (model_name == "resnet50"):
            return models.resnet50()
        if (model_name == "inception_v3"):
            return models.inception_v3()