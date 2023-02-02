import torch
import torch.nn as nn
import torch.nn.functional as F

def skew_relative(QE_t):
    B, H, N = QE_t.shape[0], QE_t.shape[1], QE_t.shape[2]
    QE_t = F.pad(QE_t, (0, 1)).view(B, H, -1)
    QE_t = F.pad(QE_t, (0, N - 1)).view(B, H, N+1, 2 * N - 1)
    return QE_t[:, :, :N, -N:]

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, head_dim, output_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.W_q = nn.Linear(input_dim, num_heads * head_dim, bias=False)
        self.W_k = nn.Linear(input_dim, num_heads * head_dim, bias=False)
        self.W_v = nn.Linear(input_dim, num_heads * head_dim, bias=False)
        self.W_o = nn.Linear(num_heads * head_dim, output_dim)

        self.scaling_factor = 1 / (head_dim ** 0.5)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lookup_table=None, mask=None):
        B = x.shape[0]

        Q = self.W_q(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # compute attention
        attention = torch.matmul(Q, K.transpose(-2, -1))
        if lookup_table is not None:
            S_rel = skew_relative(torch.matmul(Q, lookup_table.transpose(-2, -1)))
            attention += S_rel

        attention = attention * self.scaling_factor

        if mask is not None:
            attention = attention + mask

        attention = F.softmax(attention, dim=-1)
        attention = torch.matmul(attention, V).transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.head_dim)
        attention = self.W_o(attention)
        attention = self.dropout(attention)
        return attention

""" Motion In-betweening via Two-stage Transformer [Qin et al. 2022] """
class ContextTransformer(nn.Module):
    def __init__(self, motion_features, env_features, num_layers=6, num_heads=8, d_model=512, d_ff=2048):
        super(ContextTransformer, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={d_model} and num_heads={num_heads}")

        self.motion_features = motion_features
        self.env_features    = env_features
        self.num_layers      = num_layers
        self.num_heads       = num_heads
        self.d_model         = d_model
        self.d_ff            = d_ff
        
        # encoders
        self.encoder = nn.Sequential(
            nn.Linear(motion_features * 2 + env_features, d_model),
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
            nn.Linear(d_model, motion_features),
        )

    def forward(self, x, mask_in, p_kf):
        """
        Args:
            x: (B, T, D)
            mask_in: (B, T, D)
            p_kf: (B, T, 2)
        """
        B, T, D = x.shape
        device = x.device

        pe_kf = self.keyframe_pos_enc(p_kf)
        h_ctx = self.encoder(torch.cat([x, mask_in], dim=-1)) + pe_kf # (B, T, d_model)

        # relative distance range: [-T+1, ..., T-1], 2T-1 values in total
        rel_dist = torch.arange(-T+1, T, device=device, dtype=torch.float32)
        E_rel = self.relative_pos_enc(rel_dist.unsqueeze(-1)) # (2T-1, d_model)

        # m_atten: (B, num_heads, T, T)
        m_in = torch.sum(mask_in, dim=-1) # (B, T)
        m_in = torch.where(m_in > 0, 1., 0.) # (B, T)
        mask_atten = torch.zeros(B, self.num_heads, T, T, device=device, dtype=torch.float32)
        mask_atten = mask_atten.masked_fill(m_in.view(B, 1, 1, T) == 0, -1e9)

        # Transformer layers
        for i in range(len(self.layers) // 2):
            h_ctx = h_ctx + self.layers[i*2](self.layer_norm(h_ctx), lookup_table=E_rel, mask=mask_atten)
            h_ctx = h_ctx + self.layers[i*2+1](self.layer_norm(h_ctx))
        
        # decoder
        y = self.decoder(h_ctx)
        
        return y

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
        
        # encoders
        self.encoder = nn.Sequential(
            nn.Linear(dof * 2, d_model),
            nn.PReLU(),
            nn.Linear(d_model, d_model),
            nn.PReLU(),
        )
        self.relative_pos_enc = nn.Sequential(
            nn.Linear(1, d_model),
            nn.PReLU(),
            nn.Linear(d_model, d_model // num_heads),
        )

        # Transformer layers
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
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
            nn.Linear(d_model, dof + 4),
        )

    def forward(self, x, mask_in):
        """
        :param x: (B, T, D)
        :param mask_in: (B, T, D)
        """
        B, T, D = x.shape
        device = x.device

        h_ctx = self.encoder(torch.cat([x, mask_in], dim=-1))

        # relative distance range: [-T+1, ..., T-1], 2T-1 values in total
        rel_dist = torch.arange(-T+1, T, device=device, dtype=torch.float32)
        E_rel = self.relative_pos_enc(rel_dist.unsqueeze(-1)) # (2T-1, d_model)

        # m_atten: (B, num_heads, T, T)
        mask_atten = torch.zeros(B, self.num_heads, T, T, device=device, dtype=torch.float32)

        # Transformer layers
        for i in range(len(self.layers) // 2):
            h_ctx = h_ctx + self.layers[i*2](self.layer_norm(h_ctx), E_rel=E_rel, mask=mask_atten)
            h_ctx = h_ctx + self.layers[i*2+1](self.layer_norm(h_ctx))
        
        # decoder
        y = self.decoder(h_ctx)
        y, contacts = torch.split(y, [self.dof, 4], dim=-1)
        
        return y, torch.sigmoid(contacts)
