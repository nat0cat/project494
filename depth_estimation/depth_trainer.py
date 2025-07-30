import torch
import torch.nn.functional as F
from trainer import Trainer

def pose_vec2mat(vec):
    t = vec[:, :3].unsqueeze(-1)
    angle = vec[:, 3:]
    angle_norm = torch.norm(angle, dim=1, keepdim=True)
    axis = angle / (angle_norm + 1e-8)
    angle = angle_norm.unsqueeze(-1)

    K = torch.tensor([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ], dtype=torch.float32, device=vec.device)

    cross = torch.zeros(vec.shape[0], 3, 3).to(vec.device)
    cross[:, 0, 1] = -axis[:, 2]
    cross[:, 0, 2] = axis[:, 1]
    cross[:, 1, 0] = axis[:, 2]
    cross[:, 1, 2] = -axis[:, 0]
    cross[:, 2, 0] = -axis[:, 1]
    cross[:, 2, 1] = axis[:, 0]

    R = torch.eye(3, device=vec.device).unsqueeze(0) + \
        torch.sin(angle) * cross + \
        (1 - torch.cos(angle)) * torch.bmm(cross, cross)

    T = torch.cat([R, t], dim=-1)
    last_row = torch.tensor([0, 0, 0, 1], device=vec.device).view(1, 1, 4).repeat(vec.shape[0], 1, 1)
    return torch.cat([T, last_row], dim=1)


def project_3d(depth, intrinsics, pose):
    b, _, h, w = depth.shape
    grid_x, grid_y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
    ones = torch.ones_like(grid_x)
    pixels = torch.stack([grid_x, grid_y, ones], dim=0).float().to(depth.device)
    pixels = pixels.view(3, -1).unsqueeze(0).repeat(b, 1, 1)  # (B, 3, H*W)

    cam_points = torch.inverse(intrinsics).bmm(pixels) * depth.view(b, 1, -1)
    cam_points = torch.cat([cam_points, torch.ones_like(cam_points[:, :1])], dim=1)
    proj = intrinsics.bmm(pose[:, :3, :])
    coords = proj.bmm(cam_points)

    x = coords[:, 0] / (coords[:, 2] + 1e-7)
    y = coords[:, 1] / (coords[:, 2] + 1e-7)

    x = 2 * (x / w - 0.5)
    y = 2 * (y / h - 0.5)

    grid = torch.stack([x, y], dim=-1).view(b, h, w, 2)
    return grid


def photometric_loss(tgt_img, warped_img):
    return torch.abs(tgt_img - warped_img).mean()


def edge_aware_smoothness(depth, img):
    # Compute image gradients first
    img_gray = img.mean(1, keepdim=True)  # Convert to grayscale [B, 1, H, W]

    dx_depth = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    dy_depth = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

    dx_img = torch.abs(img_gray[:, :, :, :-1] - img_gray[:, :, :, 1:])
    dy_img = torch.abs(img_gray[:, :, :-1, :] - img_gray[:, :, 1:, :])

    # Apply exponential weighting
    weight_x = torch.exp(-torch.abs(dx_img))
    weight_y = torch.exp(-torch.abs(dy_img))

    loss = (dx_depth * weight_x).mean() + (dy_depth * weight_y).mean()
    return loss

class DepthTrainer(Trainer):
    def __init__(self, depth_net, pose_net, optimizer, intrinsics, train_loader, device, epochs=10):
        self.depth_net = depth_net
        self.pose_net = pose_net
        self.intrinsics = intrinsics

        super().__init__(
            model=None,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epochs=epochs)

        self.depth_net.to(self.device)
        self.pose_net.to(self.device)

    def fwd_pass(self, input_data):
        tgt, [src1, src2] = input_data
        tgt = tgt.to(self.device)
        src1 = src1.to(self.device)
        src2 = src2.to(self.device)

        disp = self.depth_net(tgt)
        depth = 1 / (disp + 1e-6)

        pose = pose_vec2mat(self.pose_net(tgt, src1))  # t -> t-1
        grid = project_3d(depth, self.intrinsics, pose)
        warped = F.grid_sample(src1, grid, padding_mode='border', align_corners=True)

        loss_photo = photometric_loss(tgt, warped)
        loss_smooth = edge_aware_smoothness(depth, tgt)
        loss = loss_photo + 0.1 * loss_smooth

        return loss
