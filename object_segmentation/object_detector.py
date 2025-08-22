import torch
import torch.nn as nn
import torch.nn.functional as F
from depth_estimation.depth_trainer import pose_vec2mat
from utils.model_loader import load_depth, load_backbone
from slot_attention import SlotAttention
from mask_decoder import MaskDecoder

# object detection module
class ObjectDetector(nn.Module):
    def __init__(self, num_slots, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.dino = load_backbone()
        self.depth_net = load_depth()
        self.slot_attn = SlotAttention(num_slots, feat_dim)
        self.mask_decoder = MaskDecoder(feat_dim, feat_dim)
        self.num_slots = num_slots

    # construct feature vector with masks and depth
    @staticmethod
    def compute_object_features(feat_map_flat, depth_flat, masks_flat):
        obj_feats = torch.einsum("bsh,bch->bsc", masks_flat, feat_map_flat)
        obj_feats = obj_feats / (masks_flat.sum(dim=-1, keepdim=True) + 1e-8)
        masked_depth = torch.einsum("bsh,bch->bsc", masks_flat, depth_flat)
        masked_depth = masked_depth / (masks_flat.sum(dim=-1, keepdim=True) + 1e-8)
        obj_feats = torch.cat([obj_feats, masked_depth], dim=-1)
        return obj_feats

    # compute confidence score from masks and bounding boxes
    @staticmethod
    def compute_confidence(masks, bboxes, min_area=5):
        B, S, H, W = masks.shape
        flat_masks = masks.view(B, S, -1)

        # activation
        avg_score = flat_masks.mean(dim=-1)
        peak_score = flat_masks.max(dim=-1).values

        # bbox area check
        widths = (bboxes[..., 2] - bboxes[..., 0]).clamp(min=0)
        heights = (bboxes[..., 3] - bboxes[..., 1]).clamp(min=0)
        box_area = widths * heights
        valid_box = (box_area > min_area).float()

        # weighted sum
        confidence = 0.6 * avg_score + 0.4 * peak_score
        confidence = confidence * valid_box

        return confidence

    # move all models to device
    def set_device(self, device):
        self.to(device)
        self.dino.to(device)
        self.depth_net.to(device)
        self.slot_attn.to(device)
        self.mask_decoder.to(device)

    # use backbone to get image features
    def extract_features(self, img):
        flat = self.dino.get_intermediate_layers(img, n=1)[0]
        B, N, C = flat.shape
        H = W = int(N ** 0.5)
        features = flat.transpose(1, 2).reshape(B, C, H, W)
        return flat, features

    # get depth
    def get_depth(self, img):
        return self.depth_net(img)

    # get pose
    def get_pose(self, img):
        return pose_vec2mat(self.pose_net(img))

    # get masks from slot attention + mask decoder
    def get_masks(self, feat, depth):
        B = feat.shape[0]
        depth_feats = torch.cat([feat, depth], dim=-1)
        slots, _ = self.slot_attn(depth_feats)
        masks = self.mask_decoder(slots, depth_feats)
        masks_flat = masks.view(B, self.num_slots, -1)
        return masks, masks_flat

    # use masks to compute bounding boxes
    def get_bounding_boxes(self, masks):
        bboxes = []
        B = masks.shape[0]
        for b in range(B):
            boxes_batch = []
            for s in range(self.num_slots):
                mask = masks[b, s]
                ys, xs = torch.where(mask > 0.5 * mask.max())
                if len(xs) == 0 or len(ys) == 0:
                    boxes_batch.append(torch.tensor([0,0,0,0], device=mask.device))
                else:
                    x1, x2 = xs.min().item(), xs.max().item()
                    y1, y2 = ys.min().item(), ys.max().item()
                    boxes_batch.append(torch.tensor([x1, y1, x2, y2], device=mask.device))
            bboxes.append(torch.stack(boxes_batch))
        bboxes = torch.stack(bboxes)
        return bboxes

    # forward pass
    def forward(self, x):
        B = x.shape[0]

        feat_map_flat, feat_map = self.extract_features(x)
        depth = self.get_depth(x)

        depth_down = F.interpolate(depth, size=(16, 16), mode='bilinear', align_corners=False)
        depth_flat = depth_down.view(B, 1, -1)

        masks, masks_flat = self.get_masks(feat_map_flat, depth_flat.permute(0, 2, 1))
        obj_feats = self.compute_object_features(feat_map_flat.view(B, 768, -1), depth_flat, masks_flat)
        bboxes = self.get_bounding_boxes(masks)
        conf = self.compute_confidence(masks, bboxes)

        return feat_map, depth, masks, bboxes, obj_feats, conf