from torchvision import models
import torch.nn as nn

def get_model(num_class = 10):
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    del model.features[7:]
    del model.features[6][1:]
    del model.avgpool
    del model.classifier
    del model.features[6][0].stochastic_depth
    del model.features[6][0].block[3]    
    model.features.append(
        nn.Sequential(
            nn.Conv2d(672, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
    )
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
    model.classifier = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, num_class)
    )
    
    return model
