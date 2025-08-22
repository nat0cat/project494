import torch
import torch.nn as nn
from scene_graph_constructor import  SceneGraphConstructor

# evolving scene graph: update node and edge embeddings across frames
class EvolvingSceneGraphTemporal(nn.Module):
    def __init__(self, node_dim=256, edge_dim=128, device='cuda'):
        super().__init__()
        # scene graph constructor
        self.sg_constructor = SceneGraphConstructor(node_dim, edge_dim, device)

        # gru to update node embeddings temporally
        self.node_gru = nn.GRUCell(node_dim, node_dim)

        # mlp to compute temporal edge embeddings
        self.temporal_edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, 256),
            nn.ReLU(),
            nn.Linear(256, edge_dim)
        )

        self.device = device
        self.prev_node_emb = None
        self.prev_edge_emb = None

    # reset previous embeddings
    def reset(self):
        self.prev_node_emb = None
        self.prev_edge_emb = None

    def forward(self, obj_feats, masks, features, bboxes, conf, depth, pose):
        # construct scene graph for current frame
        node_emb, edge_emb, adj = self.sg_constructor(obj_feats, masks, features, bboxes, conf, depth, pose)
        B, N, D = node_emb.shape

        # update nodes with temporal gru if previous exists
        if self.prev_node_emb is not None:
            node_emb_flat = node_emb.view(B * N, D)
            prev_flat = self.prev_node_emb.view(B * N, D)
            node_emb_updated = self.node_gru(node_emb_flat, prev_flat).view(B, N, D)
        else:
            node_emb_updated = node_emb

        # compute current edge embeddings
        edge_feat = self.sg_constructor.edge_features(node_emb_updated)
        edge_emb = self.sg_constructor.edge_mlp(edge_feat.view(B, N * N, -1)).view(B, N, N, -1)

        # compute temporal edges between previous nodes and current nodes
        if self.prev_node_emb is not None:
            prev_i = self.prev_node_emb.unsqueeze(2).expand(B, N, N, D)
            curr_j = node_emb_updated.unsqueeze(1).expand(B, N, N, D)
            temporal_edge_feat = torch.cat([prev_i, curr_j], dim=-1)
            temporal_edge_emb = self.temporal_edge_mlp(temporal_edge_feat.view(B, N * N, -1)).view(B, N, N, -1)
        else:
            temporal_edge_emb = None

        # store current embeddings for next frame
        self.prev_node_emb = node_emb_updated.detach()
        self.prev_edge_emb = edge_emb.detach()

        # adjacency: distance-based similarity
        adj = torch.exp(-torch.cdist(node_emb_updated, node_emb_updated))

        return node_emb_updated, edge_emb, temporal_edge_emb, adj
