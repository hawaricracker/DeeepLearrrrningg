import torch
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES, IMG_SIZE = 5, 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.convnext_tiny(weights=None)
        self.net.classifier[2] = nn.Linear(768, NUM_CLASSES)

    def forward(self, x):
        out = self.net(x)
        if not self.training:  
            out = (out + self.net(torch.flip(x, dims=[3]))) / 2
        return out


def get_model():

    return Net()


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
