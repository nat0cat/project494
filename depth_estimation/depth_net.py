import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# depth estimation network
class DepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder is constructed with a resnet18 backbone
        self.encoder = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-2])

        # decoder reconstructs image with upsampling
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    # forward pass: encoder -> decoder
    def forward(self, x):
        return self.decoder(self.encoder(x))


