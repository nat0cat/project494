import torch
import torch.nn.functional as F
from depth_utils import project_3d, pose_vec2mat

### LOSS FUNCTIONS ###

# structural similarity
def ssim(img1, img2):
    # constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # compute local mean
    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)

    # compute local variance
    sigma1 = F.avg_pool2d(img1 ** 2, 3, 1, 1) - mu1 ** 2
    sigma2 = F.avg_pool2d(img2 ** 2, 3, 1, 1) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2

    # ssim computation
    ssim_num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)            # numerator
    ssim_den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)  # denominator

    # final map in range
    ssim_map = ssim_num / (ssim_den + 1e-7)
    return torch.clamp((1 - ssim_map) / 2, 0, 1)

# photometric loss: guide warped image reconstruction with ssim and l1
def photometric_loss_ssim_l1(tgt_img, warped_img):
    # compute both losses
    ssim_loss = ssim(tgt_img, warped_img).mean(1, True)  # mean over channels
    l1_loss = torch.abs(tgt_img - warped_img).mean(1, True)

    # weighted average
    alpha = 0.85
    return (alpha * ssim_loss + (1 - alpha) * l1_loss).mean()

# photometric loss: guide accuracy of warped image reconstruction
def photometric_loss_l1(tgt_img, warped_img):
    # loss is computed as the average difference between the target image and warped image
    loss = torch.abs(tgt_img - warped_img).mean()
    return loss

# edge aware smoothness: guides image boundaries
def edge_aware_smoothness(depth, img):
    # convert to grayscale
    img_gray = img.mean(1, keepdim=True)

    # compute image gradients
    dx_depth = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    dy_depth = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
    dx_img = torch.abs(img_gray[:, :, :, :-1] - img_gray[:, :, :, 1:])
    dy_img = torch.abs(img_gray[:, :, :-1, :] - img_gray[:, :, 1:, :])

    # exponential weighting
    weight_x = torch.exp(-torch.abs(dx_img))
    weight_y = torch.exp(-torch.abs(dy_img))

    # compute loss as a weighted sum
    loss = (dx_depth * weight_x).mean() + (dy_depth * weight_y).mean()
    return loss

def forward_pass(depth_net, pose_net, input_data, device, intrinsics):
    # unpack input data
    tgt, src = input_data
    tgt = tgt.to(device)
    src = src.to(device)

    # compute depth map
    disp = depth_net(tgt)
    disp = F.interpolate(disp, size=(224, 224), mode='bilinear', align_corners=True)
    depth = 1 / (disp + 1e-6)

    # compute pose estimation
    pose = pose_vec2mat(pose_net(tgt, src))

    # construct warped image
    grid = project_3d(depth, intrinsics, pose)
    warped = F.grid_sample(src, grid, padding_mode='border', align_corners=True)

    return disp, depth, pose, warped


