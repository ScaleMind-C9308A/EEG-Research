import torch
import torch.nn as nn
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# compute the current classification accuracy
def compute_acc(preds, labels):
    """
    preds: Tensor([batch_size, num_classes])
    labels: Tensor([batch_size])
    """
    _, preds = torch.max(preds, 1)
    acc = torch.sum(preds == labels).double() / len(labels)
    acc = acc.item()
    return acc