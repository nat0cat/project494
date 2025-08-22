import torch
import torch.nn as nn


class SceneGraphConstructor(nn.Module):
    def __init__(self, node_dim=256, edge_dim=128, device='cuda'):
        super().__init__()
        self.device = device
        self.intrinsics = torch.eye(3).repeat(1, 1).to(device)

        # node mlp
        self.node_mlp = nn.Sequential(
            nn.Linear(768 + 769 + 4 + 1 + 3, 512),
            nn.ReLU(),
            nn.Linear(512, node_dim))

        # edge mlp
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, 256),
            nn.ReLU(),
            nn.Linear(256, edge_dim))

    # edge feature vector construction
    @staticmethod
    def edge_features(node_emb):
        B, N, D = node_emb.shape
        n_i = node_emb.unsqueeze(2).expand(B, N, N, D)
        n_j = node_emb.unsqueeze(1).expand(B, N, N, D)
        edge_feat = torch.cat([n_i, n_j], dim=-1)
        return edge_feat

    # pool masks and features
    @staticmethod
    def mask_pooling(features, masks):
        B, C, H, W = features.shape
        N = masks.shape[1]

        # flatten spatial dims
        features_flat = features.view(B, C, H * W)
        masks_flat = masks.view(B, N, H * W)

        # normalize mask
        masks_norm = masks_flat / (masks_flat.sum(dim=-1, keepdim=True) + 1e-6)

        # weighted sum
        pooled = torch.einsum('bnh,bch->bnc', masks_norm, features_flat)
        return pooled

    # position feature vector
    @staticmethod
    def positional_encoding(bboxes):
        # compute center
        x1, y1, x2, y2 = bboxes.unbind(-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # find dimensions
        w = x2 - x1
        h = y2 - y1

        # construct feature
        pos_feat = torch.stack([cx, cy, w, h], dim=-1)
        return pos_feat

    # position in next frame
    @staticmethod
    def motion_prior(masks, depth, pose, K):
        B, N, H, W = masks.shape
        device = masks.device

        # pixel coordinate centroid
        ys = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, N, H, W)
        xs = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, N, H, W)
        mask_sum = masks.sum([-1, -2]) + 1e-6
        cx = (masks * xs).sum([-1, -2]) / mask_sum
        cy = (masks * ys).sum([-1, -2]) / mask_sum

        # depth at centroid
        cx_idx = cx.round().long().clamp(0, W - 1)
        cy_idx = cy.round().long().clamp(0, H - 1)

        # collect
        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)
        z = depth[batch_idx, 0, cy_idx, cx_idx]

        # 3D coordinates in camera frame
        X = (cx - K[0, 2]) * z / K[0, 0]
        Y = (cy - K[1, 2]) * z / K[1, 1]
        P_3D = torch.stack([X, Y, z], dim=-1)

        # homogeneous coordinates and transform to next frame
        P_h = torch.cat([P_3D, torch.ones(B, N, 1, device=device)], dim=-1)
        P_next = torch.einsum('bij,bnj->bni', pose, P_h)
        P_next = P_next[..., :3] / P_next[..., 3:]

        return P_next

    # forward pass
    def forward(self, obj_feats, masks, features, bboxes, conf, depth, pose):
        # get features
        pooled_feats = self.mask_pooling(features, masks)
        pos_feat = self.positional_encoding(bboxes)
        motion = self.motion_prior(masks, depth, pose, self.intrinsics)

        # node construction
        node_input = torch.cat([pooled_feats, obj_feats, pos_feat, conf.unsqueeze(-1), motion], dim=-1)
        node_emb = self.node_mlp(node_input)

        # edge construction
        edge_feat = self.edge_features(node_emb)
        B, N, _, _ = edge_feat.shape
        edge_emb = self.edge_mlp(edge_feat.view(B, N * N, -1))
        edge_emb = edge_emb.view(B, N, N, -1)

        # soft adjacency matrix
        adj = torch.exp(-torch.cdist(node_emb, node_emb))

        return node_emb, edge_emb, adj
