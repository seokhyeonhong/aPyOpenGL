import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mlp import MLP, MultiLinear, SineMLP
from model.transformer import MultiHeadAttention, PhaseMultiHeadAttention

class ContextTransformer(nn.Module):
    def __init__(self, dof, num_layers=6, num_heads=8, d_model=512, d_ff=2048):
        super(ContextTransformer, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={d_model} and num_heads={num_heads}")

        self.dof = dof
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.encoder = MLP(dof+1, [d_model], d_model, activation=nn.PReLU(), activation_at_last=True)
        self.keyframe_pos_enc = MLP(2, [d_model], d_model, activation=nn.PReLU(), activation_at_last=False)
        self.relative_pos_enc = MLP(1, [d_model], d_model // num_heads, activation=nn.PReLU(), activation_at_last=False)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-8)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(MultiHeadAttention(512, head_dim=d_model // num_heads, output_dim=d_model, num_heads=num_heads, dropout=0))
            self.layers.append(MLP(d_model, [d_ff], d_model, activation=nn.PReLU(), activation_at_last=False))
        self.decoder = MLP(d_model, [d_model], dof + 4, activation=nn.PReLU(), activation_at_last=False)

    def forward(self, x, mask_in, p_kf):
        pe_kf = self.keyframe_pos_enc(p_kf)
        h_ctx = self.encoder(torch.cat([x, mask_in], dim=-1)) + pe_kf

        E_rel = self.relative_pos_enc(torch.arange(-h_ctx.shape[1] + 1, h_ctx.shape[1], device=h_ctx.device, dtype=torch.float32)[None, :, None])
        mask_atten = torch.zeros(x.shape[0], self.num_heads, x.shape[1], x.shape[1], device=h_ctx.device, dtype=torch.float32)
        mask_atten = mask_atten.masked_fill(mask_in.view(x.shape[0], 1, 1, x.shape[1]) == 0, -1e8)

        for i in range(len(self.layers) // 2):
            h_ctx = h_ctx + self.layers[i*2](self.layer_norm(h_ctx), E_rel=E_rel, mask=mask_atten)
            h_ctx = h_ctx + self.layers[i*2+1](self.layer_norm(h_ctx))
        y = self.decoder(h_ctx)
        y, contacts = torch.split(y, [self.dof, 4], dim=-1)
        return y, torch.sigmoid(contacts)

class DetailTransformer(nn.Module):
    def __init__(self, dof, num_layers=6, num_heads=8, d_model=512, d_ff=2048):
        super(DetailTransformer, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={d_model} and num_heads={num_heads}")

        self.dof = dof
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.encoder = MLP(dof+1, [d_model], d_model, activation=nn.PReLU(), activation_at_last=True)
        self.relative_pos_enc = MLP(1, [d_model], d_model // num_heads, activation=nn.PReLU(), activation_at_last=False)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-8)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(MultiHeadAttention(d_model, head_dim=d_model // num_heads, output_dim=d_model, num_heads=num_heads, dropout=0))
            self.layers.append(MLP(d_model, [d_ff], d_model, activation=nn.PReLU(), activation_at_last=False))
        self.decoder = MLP(d_model, [d_model], dof + 4, activation=nn.PReLU(), activation_at_last=False)

    def forward(self, x, mask_in):
        h_ctx = self.encoder(torch.cat([x, mask_in], dim=-1))

        E_rel = self.relative_pos_enc(torch.arange(-h_ctx.shape[1] + 1, h_ctx.shape[1], device=h_ctx.device, dtype=torch.float32)[None, :, None])
        mask_atten = torch.zeros(x.shape[0], self.num_heads, x.shape[1], x.shape[1], device=h_ctx.device, dtype=torch.float32)

        for i in range(len(self.layers) // 2):
            h_ctx = h_ctx + self.layers[i*2](self.layer_norm(h_ctx), E_rel=E_rel, mask=mask_atten)
            h_ctx = h_ctx + self.layers[i*2+1](self.layer_norm(h_ctx))
        y = self.decoder(h_ctx)
        y, contacts = torch.split(y, [self.dof, 4], dim=-1)
        return y, torch.sigmoid(contacts)

class DetailTransformerResidual(nn.Module):
    def __init__(self, dof, num_layers=6, num_heads=8, d_model=512, d_ff=2048):
        super(DetailTransformerResidual, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={d_model} and num_heads={num_heads}")

        self.dof = dof
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff

        self.encoder = MLP(dof+1, [d_model], d_model, activation=nn.PReLU(), activation_at_last=True)
        self.relative_pos_enc = MLP(1, [d_model], d_model // num_heads, activation=nn.PReLU(), activation_at_last=False)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-8)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(MultiHeadAttention(d_model, head_dim=d_model // num_heads, output_dim=d_model, num_heads=num_heads, dropout=0))
            self.layers.append(MLP(d_model, [d_ff], d_model, activation=nn.PReLU(), activation_at_last=False))
        self.decoder = MLP(d_model, [d_model], dof + 4, activation=nn.PReLU(), activation_at_last=False)

    def forward(self, x, mask_in):
        h_ctx = self.encoder(torch.cat([x, mask_in], dim=-1))

        E_rel = self.relative_pos_enc(torch.arange(-h_ctx.shape[1] + 1, h_ctx.shape[1], device=h_ctx.device, dtype=torch.float32)[None, :, None])
        mask_atten = torch.zeros(x.shape[0], self.num_heads, x.shape[1], x.shape[1], device=h_ctx.device, dtype=torch.float32)

        for i in range(len(self.layers) // 2):
            h_ctx = h_ctx + self.layers[i*2](self.layer_norm(h_ctx), E_rel=E_rel, mask=mask_atten)
            h_ctx = h_ctx + self.layers[i*2+1](self.layer_norm(h_ctx))
        y = self.decoder(h_ctx)
        y, contacts = torch.split(y, [self.dof, 4], dim=-1)
        return y + x, torch.sigmoid(contacts)

class PhaseTransformer(nn.Module):
    def __init__(self, dof, num_layers=6, num_heads=8, d_model=512, d_ff=2048):
        super(PhaseTransformer, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={d_model} and num_heads={num_heads}")

        self.dof = dof
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff

        self.encoder = MLP(dof+1, [d_model], d_model, activation=nn.PReLU(), activation_at_last=True)
        self.relative_pos_enc = MLP(1, [d_model], d_model // num_heads, activation=nn.PReLU(), activation_at_last=False)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-8)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(PhaseMultiHeadAttention(d_model, head_dim=d_model // num_heads, output_dim=d_model, num_heads=num_heads, dropout=0))
            self.layers.append(MLP(d_model, [d_ff], d_model, activation=nn.PReLU(), activation_at_last=False))
        self.decoder = MLP(d_model, [d_model], dof + 4, activation=nn.PReLU(), activation_at_last=False)

    def forward(self, x, mask_in, phase):
        h_ctx = self.encoder(torch.cat([x, mask_in], dim=-1))

        E_rel = self.relative_pos_enc(torch.arange(-h_ctx.shape[1] + 1, h_ctx.shape[1], device=h_ctx.device, dtype=torch.float32)[None, :, None])
        mask_atten = torch.zeros(x.shape[0], self.num_heads, x.shape[1], x.shape[1], device=h_ctx.device, dtype=torch.float32)

        for i in range(len(self.layers) // 2):
            h_ctx = h_ctx + self.layers[i*2](self.layer_norm(h_ctx), phase.transpose(1, 2).unsqueeze(-1), E_rel=E_rel, mask=mask_atten)
            h_ctx = h_ctx + self.layers[i*2+1](self.layer_norm(h_ctx))
        y = self.decoder(h_ctx)
        y, contacts = torch.split(y, [self.dof, 4], dim=-1)
        return y, torch.sigmoid(contacts)

class GatingNetwork(nn.Module):
    def __init__(self, dof, d_tta=32, num_experts=8):
        super(GatingNetwork, self).__init__()
        self.enc = MLP(dof, [512], 512, activation=nn.PReLU(), activation_at_last=True)
        self.layers = MLP(512, [512, 512], num_experts, activation=nn.PReLU(), activation_at_last=False)
        self.tta_emb = TimeToArrival(d_tta)
        self.tta_enc = MLP(d_tta, [512, 512], 512, activation=nn.PReLU(), activation_at_last=False)
        
    def forward(self, x):
        tta = torch.arange(x.shape[1]-1, -1, step=-1, device=x.device, dtype=torch.float32)
        pe_tta = self.tta_enc(self.tta_emb(tta)).expand(x.shape[0], -1, -1)
        h_enc = self.enc(x) + pe_tta
        y = self.layers(h_enc)
        y = F.softmax(y, dim=-1)
        return y

class MultiPhaseDecoder(nn.Module):
    def __init__(self, dof, num_experts=4):
        super(MultiPhaseDecoder, self).__init__()
        self.layers = nn.Sequential(
            MultiLinear(num_experts, dof, 512, True),
            nn.PReLU(),
            MultiLinear(num_experts, 512, 512, True),
            nn.PReLU(),
            MultiLinear(num_experts, 512, dof, True),
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.layers(x)

class TimeToArrival(nn.Module):
    def __init__(self, dim):
        super(TimeToArrival, self).__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, but dim={dim}")
        self.dim = dim
        self.div_term = nn.Parameter(1.0 / torch.pow(10000, torch.arange(0, dim, step=2) / dim))

    def forward(self, tta):
        tta = tta.unsqueeze(1)
        embedding = torch.empty((tta.shape[0], self.dim), device=tta.device, dtype=torch.float32)
        embedding[:, 0::2] = torch.sin(tta * self.div_term)
        embedding[:, 1::2] = torch.cos(tta * self.div_term)
        return embedding