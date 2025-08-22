import torch
import torch.nn as nn

# mask decoder
class MaskDecoder(nn.Module):
    def __init__(self, feat_dim: int, slot_dim: int):
        super().__init__()
        self.slot_proj = nn.Linear(slot_dim, feat_dim+1)

    def forward(self, slots: torch.Tensor, feat_map: torch.Tensor) -> torch.Tensor:
        B,S,D = slots.shape
        B,C,H,W = (4, 768, 16, 16)

        q = self.slot_proj(slots)
        attn = torch.einsum('bsc,bnc->bsn', q, feat_map)
        attn = attn.softmax(dim=1)
        masks = attn.view(B,S,H,W)
        return masks