import torch
import torch.nn as nn

# pose estimation network
class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 6)
        )

    # forward pass
    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        return self.net(x)