import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# depth encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # resnet as a backbone
        res_encoder = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-2])

        # layers of resnet to extract skip connections
        self.b0 = nn.Sequential(
            res_encoder[0],
            res_encoder[1],
            res_encoder[2])
        self.mp = res_encoder[3]
        self.b1 = res_encoder[4]
        self.b2 = res_encoder[5]
        self.b3 = res_encoder[6]
        self.b4 = res_encoder[7]

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.mp(x0)
        x1 = self.b1(x1)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x4 = self.b4(x3)
        return x0, x1, x2, x3, x4


# depth encoder layer
class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # learnable upsampling
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # conv block after concatenation with skip
        self.conv = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)),
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels)),
                nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# depth estimation network
class DepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder is constructed with a resnet18 backbone
        self.encoder = Encoder()

        # decoder reconstructs image with upsampling
        self.dec3 = Decoder(512, 256, 256)
        self.dec2 = Decoder(256, 128, 128)
        self.dec1 = Decoder(128, 64, 64)
        self.dec0 = Decoder(64, 64, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    # forward pass: encoder -> decoder
    def forward(self, x):
        # pass thru decoder layers
        x0, x1, x2, x3, x4 = self.encoder(x)
        d3 = self.dec3(x4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        d0 = self.dec0(d1, x0)

        # final upsample to input size
        out = nn.functional.interpolate(d0, size=x.shape[2:], mode="bilinear", align_corners=False)
        out = self.final_conv(out)
        return out