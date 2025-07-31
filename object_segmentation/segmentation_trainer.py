import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.trainer import Trainer

### HELPER FUNCTIONS ###

# uses a pretrained backbone to extract features
def extract_features(extractor, image):
    with torch.no_grad():
        image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        image = image * 2 - 1
        output = extractor.forward_features(image)
        patch_tokens = output["x_norm_patchtokens"]
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        features = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
    return features

# returns a depth map using a depth estimation network
def get_depth_map(depth_net, img):
    with torch.no_grad():
        depth_map = depth_net(img)
    return depth_map

# detects edges with depth
def compute_depth_edges(depth):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=depth.device).view(1,1,3,3)
    sobel_y = sobel_x.transpose(2,3)

    grad_x = F.conv2d(depth, sobel_x, padding=1)
    grad_y = F.conv2d(depth, sobel_y, padding=1)
    edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    return edges

### LOSS FUNCTIONS ###

def reconstruction_loss(recon, features):
    target = features.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    target = target.unsqueeze(1)
    target_exp = target.expand_as(recon)
    mse_loss = nn.MSELoss()
    loss = mse_loss(recon, target_exp)
    return loss

def unmasked_loss(masks, features):
    unmasked = (1 - masks) * features
    loss = torch.mean(unmasked ** 2)
    return loss

def entropy_loss(masks):
    entropy = -masks * torch.log(masks + 1e-8)
    loss = entropy.mean()
    return loss

def orthogonality_loss(masks):
    B, K, H, W = masks.shape
    masks_flat = masks.view(B, K, -1)
    gram = torch.bmm(masks_flat, masks_flat.transpose(1, 2))
    eye = torch.eye(K).to(device).unsqueeze(0)
    loss = F.mse_loss(gram / H / W, eye.expand_as(gram))
    return loss

def competition_loss(masks):
    sum_masks = masks.sum(dim=1, keepdim=True)
    competition = masks / (sum_masks + 1e-8)
    loss = -torch.mean(competition * torch.log(competition + 1e-8))
    return loss

def depth_edge_loss(masks, depth):
    edges = compute_depth_edges(depth)

    _, _, H, W = masks.shape
    edges = F.interpolate(edges, size=(H, W), mode='bilinear', align_corners=False)

    edges = edges.expand_as(masks)

    mask_grad_x = masks[:, :, :, 1:] - masks[:, :, :, :-1]
    mask_grad_y = masks[:, :, 1:, :] - masks[:, :, :-1, :]

    edge_x = edges[:, :, :, 1:]
    edge_y = edges[:, :, 1:, :]

    loss_x = ((1 - edge_x) * mask_grad_x.abs()).mean()
    loss_y = ((1 - edge_y) * mask_grad_y.abs()).mean()

    return loss_x + loss_y

### SEGMENTATION TRAINER ###
class SegmentationTrainer(Trainer):
    def __init__(self, segmentation_net, optimizer, train_loader, device, depth_net=None, dino=None, epochs=100):

        # initialize parameters in general trainer class
        super().__init__(
            model=segmentation_net,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epochs=epochs)

        if dino is None:
            self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
        else:
            self.dino = dino

        if depth_net is not None:
            self.depth_net = depth_net.to(device)
            self.depth_net.eval()
        else:
            self.depth_net = depth_net

        self.model.to(device)
        self.dino.eval()

    def fwd_pass(self, input_data):

        # unpack input data
        tgt, [src1, src2] = input_data
        tgt = tgt.to(self.device)
        src1 = src1.to(self.device)
        src2 = src2.to(self.device)

        # extract features
        features = extract_features(self.dino, tgt)
        B, C, H, W = features.shape
        K = self.model.get_num_masks()

        # predict masks
        masks = self.model(features)

        # resize
        features_exp = features.unsqueeze(1)
        masks_exp = masks.unsqueeze(2)

        # reconstruct
        recon = masks_exp * features_exp

        # compute object detection loss
        loss_recon = reconstruction_loss(recon, features)
        loss_unmasked = unmasked_loss(masks_exp, features_exp)
        loss_entropy = entropy_loss(masks)
        loss_orthog = orthogonality_loss(masks)
        loss_comp = competition_loss(masks)
        loss = loss_recon + 0.1 * loss_unmasked + 0.01 * loss_entropy + 0.05 * loss_orthog + 0.01 * loss_comp

        # if training with pretrained depth estimation
        if self.depth_net is not None:
            depth_map = get_depth_map(self.depth_net, tgt)
            loss_depth = depth_edge_loss(masks, depth_map)
            loss += 0.2 * loss_depth

        return loss