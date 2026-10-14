import torch.nn as nn
from torchvision import transforms

NUM_CLASSES, IMG_SIZE = 5, 128
# 0 botol_plastik · 1 gelas_plastik · 2 kertas_kardus · 3 saset_kemasan · 4 styrofoam


class SiDSCConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels), nn.BatchNorm2d(in_channels), nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels), nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.network(x)
        out = out + self.proj(x)
        return self.relu(out)


class DSCNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.linear = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))
        self.network = nn.Sequential(
            SiDSCConv(3, 64), nn.MaxPool2d(2),
            SiDSCConv(64, 128), nn.MaxPool2d(2),
            SiDSCConv(128, 256), nn.MaxPool2d(2),
            SiDSCConv(256, 512), nn.MaxPool2d(2),
            SiDSCConv(512, 512),
        )

    def forward(self, x):
        return self.linear(self.network(x))


def get_model():
    """Arsitektur saja, tanpa memuat bobot."""
    return DSCNet()


def get_transform():
    return transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
