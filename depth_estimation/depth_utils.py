import torch

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

# set image range back to [0, 1] for loss calculations
def standardize_range(img):
    # reverse of initial transformation tensors used
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device)

    # back to [0, 1]
    return img * imagenet_std + imagenet_mean