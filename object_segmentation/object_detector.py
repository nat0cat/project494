import torch.nn as nn

# object segmentation module
class ObjSegmentation(nn.Module):
    # intialize
    def __init__(self, feature_dim, num_masks=10):
        super().__init__()

        # store the number of masks (10 by default)
        self.num_masks = num_masks

        # layers
        self.block1 = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.ReLU())
        self.block2 = nn.Sequential(
            nn.Conv2d(128, num_masks, kernel_size=1),
            nn.Sigmoid())

    # forward pass
    def forward(self, features):
        x = self.block1(features)
        masks = self.block2(x)
        return masks

    # returns number of masks
    def get_num_masks(self):
        return self.num_masks