import torch
import torch.nn as nn
import torch.nn.functional as F

from pymovis.utils import torchconst
from pymovis.learning.transformer import MultiHeadAttention

class KinematicTransformer(nn.Module):
    def __init__(self, geometry_dim, motion_dim, num_layers=6, num_heads=8, d_model=512, d_ff=2048):
        super(KinematicTransformer, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={d_model} and num_heads={num_heads}")

        self.geometry_dim = geometry_dim
        self.motion_dim = motion_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        
        # encoders
        self.encoder = nn.Sequential(
            nn.Linear(self.geometry_dim + self.motion_dim * 2, d_model), # geometry, motion, motion mask
            nn.PReLU(),
            nn.Linear(d_model, d_model),
            nn.PReLU(),
        )
        self.keyframe_pos_enc = nn.Sequential(
            nn.Linear(2, d_model),
            nn.PReLU(),
            nn.Linear(d_model, d_model),
        )
        self.relative_pos_enc = nn.Sequential(
            nn.Linear(1, d_model),
            nn.PReLU(),
            nn.Linear(d_model, d_model // num_heads),
        )
        
        # Transformer layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(MultiHeadAttention(d_model, head_dim=d_model // num_heads, output_dim=d_model, num_heads=num_heads, dropout=0))
            self.layers.append(nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.PReLU(),
                nn.Linear(d_ff, d_model),
            ))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.PReLU(),
            nn.Linear(d_model, self.motion_dim),
        )

    def forward(self, x, mask, kf_pos):
        B, T, D = x.shape
        device  = x.device

        kf_pos_emb = self.keyframe_pos_enc(kf_pos)
        h_ctx      = self.encoder(torch.cat([x, mask], dim=-1)) + kf_pos_emb # (B, T, d_model)

        # relative distance range: [-T+1, ..., T-1], 2T-1 values in total
        rel_dist = torch.arange(-T+1, T, device=device, dtype=torch.float32)
        E_rel    = self.relative_pos_enc(rel_dist.unsqueeze(-1)) # (2T-1, d_model)

        # Transformer layers
        for i in range(len(self.layers) // 2):
            h_ctx = h_ctx + self.layers[i*2](self.layer_norm(h_ctx), E_rel=E_rel, mask=None)
            h_ctx = h_ctx + self.layers[i*2+1](self.layer_norm(h_ctx))
        
        # decoder
        y = self.decoder(h_ctx)

        return y