import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNTracker(nn.Module):
    def __init__(self, node_dim=256, edge_dim=128, hidden_dim=256):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.temp_proj = nn.Linear(edge_dim, hidden_dim)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, node_emb, edge_emb, temporal_edge_emb=None):
        B, N, _ = node_emb.shape

        # project
        node_h = self.node_proj(node_emb)
        edge_h = self.edge_proj(edge_emb)

        if temporal_edge_emb is not None:
            temp_h = self.temp_proj(temporal_edge_emb)
            edge_h = edge_h + temp_h

        # aggregate messages
        node_exp = node_h.unsqueeze(1).expand(B, N, N, -1)
        messages = edge_h * node_exp
        messages = messages.sum(dim=2)

        # update nodes with GRU
        node_h = self.update(messages.reshape(-1, messages.size(-1)),
                             node_h.reshape(-1, node_h.size(-1)))
        node_h = node_h.view(B, N, -1)

        return node_h, edge_h
