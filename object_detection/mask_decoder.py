import torch
import torch.nn as nn

# mask decoder
class MaskDecoder(nn.Module):
    def __init__(self, feat_dim: int, slot_dim: int):
        super().__init__()
        # project slots to feature space
        self.slot_proj = nn.Linear(slot_dim, feat_dim+1)

    def forward(self, slots: torch.Tensor, feat_map: torch.Tensor) -> torch.Tensor:
        # get shapes
        B,S,D = slots.shape
        B,C,H,W = (4, 768, 16, 16)

        # project slots
        q = self.slot_proj(slots)

        # compute attention
        attn = torch.einsum('bsc,bnc->bsn', q, feat_map)
        attn = attn.softmax(dim=1)

        # reshape to spatial masks
        masks = attn.view(B,S,H,W)
        return masks