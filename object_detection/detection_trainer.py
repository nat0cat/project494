import torch
import torch.nn.functional as F

# mask reconstruction: guide slots to reconstruct feature map
def mask_reconstruction_loss(feat_map_flat, masks_flat, slot_feats):
    B, S, HW = masks_flat.shape
    _, C, _ = feat_map_flat.shape

    # weighted sum of slots using masks
    recon = torch.einsum('bsh,bsc->bch', masks_flat, slot_feats)

    # mse between reconstruction and original features
    loss = F.mse_loss(recon, feat_map_flat)
    return loss


# mask entropy: encourage masks to be confident (0 or 1)
def mask_entropy_loss(masks_flat):
    p = masks_flat.clamp(1e-6, 1-1e-6)
    loss = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return loss.mean()


# depth edge alignment: encourage masks to align with depth edges
def depth_edge_loss(masks_flat, depth_map):
    # compute depth gradients
    grad_x = torch.abs(depth_map[:, :, :, 1:] - depth_map[:, :, :, :-1])
    grad_y = torch.abs(depth_map[:, :, 1:, :] - depth_map[:, :, :-1, :])

    # downsample gradients to mask resolution
    grad_x_down = F.interpolate(grad_x, size=(16, 16), mode='bilinear', align_corners=False)
    grad_y_down = F.interpolate(grad_y, size=(16, 16), mode='bilinear', align_corners=False)

    # sum edges
    depth_edges = grad_x_down + grad_y_down

    # flatten edges
    depth_edges_flat = depth_edges.view(depth_edges.size(0), 1, -1)

    # compute weighted sum with masks
    loss = torch.einsum('bsh,bch->bsc', masks_flat, depth_edges_flat).mean()
    return loss


# mask orthogonality: encourage masks to be distinct
def mask_orthogonality_loss(masks_flat):
    B, S, HW = masks_flat.shape
    loss = 0.0
    for b in range(B):
        mask = masks_flat[b]

        # dot product between masks
        dot = torch.matmul(mask, mask.T)

        # remove self-dot terms
        dot = dot - torch.diag(torch.diag(dot))

        # average off-diagonal terms
        loss += dot.sum() / (S*(S-1))
    return loss / B

# object detection loss: combine losses
def object_detection_loss(tgt, values, lambda_recon=1.0,lambda_entropy=0.1,lambda_ortho=0.1,lambda_depth=0.1):
    # unpack values
    features, depth_map, slots, masks, _, _, _ = values
    B = features.shape[0]

    # compute flat values
    masks_flat = masks.view(B, 10, -1)
    feat_map_flat = features.view(B, 768, -1)

    # compute losses
    loss = lambda_recon * mask_reconstruction_loss(feat_map_flat, masks_flat, slots)
    loss += lambda_entropy * mask_entropy_loss(masks_flat)
    loss += lambda_ortho * mask_orthogonality_loss(masks_flat)
    loss += lambda_depth * depth_edge_loss(masks_flat, depth_map)
    return loss
