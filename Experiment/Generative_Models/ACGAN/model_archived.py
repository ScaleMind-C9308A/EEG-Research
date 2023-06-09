import torch.nn as nn
from Encoder.eeg_encoder import load_eeg_encoder
from Encoder.image_encoder import load_image_encoder
def weights_init(m):
    """custom weights initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, z_dim, g_hidden, image_channel):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.g_hidden = g_hidden
        self.image_channel = image_channel
        self.main = nn.Sequential(
            # 1st layer
            nn.ConvTranspose2d(self.z_dim, self.g_hidden * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.g_hidden * 8),
            nn.ReLU(True),
            # 2nd layer
            nn.ConvTranspose2d(self.g_hidden * 8, self.g_hidden * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.g_hidden * 4),
            nn.ReLU(True),
            # 3rd layer
            nn.ConvTranspose2d(self.g_hidden * 4, self.g_hidden * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.g_hidden * 2),
            nn.ReLU(True),
            # 4th layer
            nn.ConvTranspose2d(self.g_hidden * 2, self.g_hidden, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.g_hidden),
            nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(self.g_hidden, self.image_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, d_hidden, image_channel):
        super(Discriminator, self).__init__()
        self.image_channel = image_channel
        self.d_hidden = d_hidden
        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(self.image_channel, self.d_hidden, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(self.d_hidden, self.d_hidden * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.d_hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(self.d_hidden * 2, self.d_hidden * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.d_hidden * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(self.d_hidden * 4, self.d_hidden * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.d_hidden * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(self.d_hidden * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)
class EEGClassificationNet(nn.Module):
    def __init__(self, backbone_name, embedding_dim, num_classes):
        super().__init__()
        self.backbone = load_eeg_encoder(backbone_name, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, num_classes)
            # # For nn.CrossEntropyLoss() => The input is expected to contain the unnormalized logits for each class
            # nn.LogSoftmax(dim=1)
        )
    def forward(self, eeg):
        return self.backbone(eeg)
    def get_eeg_embedding(self, eeg):
        return self.backbone(eeg)
class ImageClassificationNet(nn.Module):
    def __init__(self, backbone_name, num_classes=40, feature_extract=True, use_pretrained=True):
        """
        Args:
            backbone_name: "resnet50" | "inception_v3" | ...
        """
        super().__init__()
        self.is_inception = (backbone_name=="inception_v3")
        self.backbone = load_image_encoder(backbone_name, num_classes, feature_extract, use_pretrained)
        # # For nn.CrossEntropyLoss() => The input is expected to contain the unnormalized logits for each class
        # self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, img):
        output = self.backbone(img)
        if (self.is_inception and self.training): #check if the model is in training mode
            output = output.logits
        # output = self.softmax(output)
        return output
    
def load_model(embedding_dim, num_classes=40, eeg_encoder_name="EEGChannelNet", img_encoder_name="inception_v3"):
    eeg_model = EEGClassificationNet(eeg_encoder_name, embedding_dim,num_classes)
    img_model = ImageClassificationNet(img_encoder_name, num_classes)
    netG = Generator(eeg_model, img_model)
    netD = Discriminator(img_model)
    return netG, netD

