import torch
from depth_estimation.depth_net import DepthNet
from depth_estimation.pose_net import PoseNet

def load_depth():
    depth = DepthNet()
    depth.load_state_dict(torch.load("depth_estimation/weights/depth_B.pth", map_location=torch.device('cpu')))
    depth.eval()
    return depth

def load_pose():
    pose = PoseNet()
    pose.load_state_dict(torch.load("depth_estimation/weights/pose_B.1_weights.pth", map_location=torch.device('cpu')))
    pose.eval()
    return pose

def load_backbone():
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    backbone.eval()
    return backbone