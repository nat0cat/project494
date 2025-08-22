import torch
import torch.nn as nn

# slot attention module with fixed S=10 slots
class SlotAttention(nn.Module):
    def __init__(self, num_slots: int, dim: int, iters: int = 3, mlp_hidden: int = 256):
        super().__init__()
        # fixed variables
        self.num_slots = num_slots
        self.iters = iters
        self.scale = dim ** -0.5

        # computation
        self.avg = nn.Parameter(torch.randn(1, 1, dim))
        self.logvar = nn.Parameter(torch.zeros(1, 1, dim))

        # attention layers
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim+1, dim, bias=False)
        self.to_v = nn.Linear(dim+1, dim, bias=False)

        # refinement layers
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.ReLU(inplace=True), nn.Linear(mlp_hidden, dim)
        )

        # norm layers
        self.norm_inputs = nn.LayerNorm(dim+1)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    # initialize slots with noise
    def intialize_slots(self, B):
        mu = self.avg.expand(B, self.num_slots, -1)
        sigma = self.logvar.exp().expand(B, self.num_slots, -1)
        noise = torch.randn_like(mu)
        return mu + sigma * noise

    # attention computation
    def compute_attn(self, q, k):
        logits = torch.einsum('bsc,bnc->bsn', q, k) * self.scale
        attn = logits.softmax(dim=1) + 1e-8
        attn = attn / attn.sum(dim=2, keepdim=True)
        return attn

    def forward(self, feats: torch.Tensor):
        B,N,C_in = feats.shape
        C = self.to_q.out_features

        # initialize slots
        slots = self.intialize_slots(B)

        # keys, values
        k = self.to_k(self.norm_inputs(feats))
        v = self.to_v(feats)

        # attention
        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))
            attn = self.compute_attn(q, k)
            updates = torch.einsum('bsn,bnc->bsc', attn, v)

            slots = self.gru(updates.reshape(-1, C), slots_prev.reshape(-1, C)).view(B,-1,C)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots