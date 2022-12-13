import torch
import torch.nn as nn
import torch.nn.functional as F

from pymovis.motion.utils import torchconst
from model.transformer import MultiHeadAttention

class KinematicTransformer(nn.Module):
    def __init__(self, dof, num_layers=6, num_heads=8, d_model=512, d_ff=2048):
        super(KinematicTransformer, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={d_model} and num_heads={num_heads}")

        self.dof = dof
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        
        # encoders
        num_joints = (self.dof - 3) // 6
        self.encoder = nn.Sequential(
            nn.Linear(self.dof + num_joints + 3, d_model), # dof, joints, target position
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
            nn.Linear(d_model, dof),
        )

    def forward(self, x, ik_mask, target_position, p_kf):
        """
        :param x: (B, T, D)
        :param ik_mask: (B, T, D)
        :param p_kf: (B, T, 2)
        """
        B, T, D = x.shape
        device = x.device

        pe_kf = self.keyframe_pos_enc(p_kf)
        h_ctx = self.encoder(torch.cat([x, ik_mask, target_position], dim=-1)) + pe_kf # (B, T, d_model)

        # relative distance range: [-T+1, ..., T-1], 2T-1 values in total
        rel_dist = torch.arange(-T+1, T, device=device, dtype=torch.float32)
        E_rel = self.relative_pos_enc(rel_dist.unsqueeze(-1)) # (2T-1, d_model)

        # Transformer layers
        for i in range(len(self.layers) // 2):
            h_ctx = h_ctx + self.layers[i*2](self.layer_norm(h_ctx), E_rel=E_rel, mask=None)
            h_ctx = h_ctx + self.layers[i*2+1](self.layer_norm(h_ctx))
        
        # decoder
        y = self.decoder(h_ctx)

        # return residual
        return y + x