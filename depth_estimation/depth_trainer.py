import torch
import torch.nn.functional as F
from utils.trainer import Trainer

### HELPER FUNCTIONS ###

# converts 6D pose vector into a transformation matrix (rotation + translation)
def pose_vec2mat(vec):
    # extract translation vector
    t = vec[:, :3].unsqueeze(-1)

    # extract rotation vector
    angle = vec[:, 3:]

    # normalize to compute values
    angle_norm = torch.norm(angle, dim=1, keepdim=True)
    axis = angle / (angle_norm + 1e-8)  # rotation axis
    angle = angle_norm.unsqueeze(-1)    # angle

    # compute cross product matrices for each batch
    cross = torch.zeros(vec.shape[0], 3, 3).to(vec.device)
    cross[:, 0, 1] = -axis[:, 2]
    cross[:, 0, 2] = axis[:, 1]
    cross[:, 1, 0] = axis[:, 2]
    cross[:, 1, 2] = -axis[:, 0]
    cross[:, 2, 0] = -axis[:, 1]
    cross[:, 2, 1] = axis[:, 0]

    # rotation matrix calculation
    R = torch.eye(3, device=vec.device).unsqueeze(0) + \
        torch.sin(angle) * cross + \
        (1 - torch.cos(angle)) * torch.bmm(cross, cross)

    # arbitrary bottom row for matrix
    btm = torch.tensor([0, 0, 0, 1], device=vec.device).view(1, 1, 4).repeat(vec.shape[0], 1, 1)

    # construct final 4x4 transformation matrix
    T = torch.cat([R, t], dim=-1)                    # concatenate rotation + translation -> 3x4
    transformation_mtx = torch.cat([T, btm], dim=1)  # concatenate bottom row -> 4x4

    return transformation_mtx

# projects pixels from one frame into another view using depth map and camera pose
def project_3d(depth, intrinsics, pose):
    # extract values from depth map
    b, _, h, w = depth.shape

    # create pixel grid
    grid_x, grid_y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')

    # stack into homogenous coordinates (x, y, 1)
    ones = torch.ones_like(grid_x)
    pixels = torch.stack([grid_x, grid_y, ones], dim=0).float().to(depth.device)

    # flatten pixel grid -> (B, 3, H*W)
    pixels = pixels.view(3, -1).unsqueeze(0).repeat(b, 1, 1)

    # convert to 3D camera coordinates
    cam_points = torch.inverse(intrinsics).bmm(pixels) * depth.view(b, 1, -1)

    # convert to homogenous 3D points -> (B, 4, H*W)
    cam_points = torch.cat([cam_points, torch.ones_like(cam_points[:, :1])], dim=1)

    # compute full projection matrix and project 3D points into new view
    proj = intrinsics.bmm(pose[:, :3, :])
    coords = proj.bmm(cam_points)

    # convert to pixel coordinates using homogenous normalization
    x = coords[:, 0] / (coords[:, 2] + 1e-7)
    y = coords[:, 1] / (coords[:, 2] + 1e-7)

    # normalize coordinates to [-1, 1] range
    x = 2 * (x / w - 0.5)
    y = 2 * (y / h - 0.5)

    # construct grid of warped coordinates
    grid = torch.stack([x, y], dim=-1).view(b, h, w, 2)

    return grid


### LOSS FUNCTIONS ###

# photometric loss: guide accuracy of warped image reconstruction
def photometric_loss(tgt_img, warped_img):
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


### DEPTH TRAINER CLASS ###
class DepthTrainer(Trainer):
    def __init__(self, depth_net, pose_net, optimizer, intrinsics, train_loader, device, epochs=10):
        self.depth_net = depth_net    # depth network
        self.pose_net = pose_net      # pose network
        self.intrinsics = intrinsics  # camera intrinsics

        # initialize parameters in general trainer class
        super().__init__(
            model=None,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epochs=epochs)

        # move the models to device for consistency
        self.depth_net.to(self.device)
        self.pose_net.to(self.device)

    # forward pass implementation for depth + pose network
    def fwd_pass(self, input_data):
        # unpack input data
        tgt, [src1, src2] = input_data
        tgt = tgt.to(self.device)
        src1 = src1.to(self.device)
        src2 = src2.to(self.device)

        # compute depth map
        disp = self.depth_net(tgt)
        depth = 1 / (disp + 1e-6)

        # compute pose estimation
        pose = pose_vec2mat(self.pose_net(tgt, src1))

        # construct warped image
        grid = project_3d(depth, self.intrinsics, pose)
        warped = F.grid_sample(src1, grid, padding_mode='border', align_corners=True)

        # loss calculations
        loss_photo = photometric_loss(tgt, warped)
        loss_smooth = edge_aware_smoothness(depth, tgt)
        loss = loss_photo + 0.1 * loss_smooth

        # pass loss into training loop in trainer class
        return loss
