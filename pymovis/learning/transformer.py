import torch
import torch.nn as nn
import torch.nn.functional as F

def skew(QE_t):
    """ Used for relative positional encoding. Implementation from Music Transformer [Huang et al. 2018] """
    B, H, T, _ = QE_t.shape # (B, H, T, 2T-1)

    QE_t = F.pad(QE_t, (0, 1)).view(B, H, 2*T*T)
    QE_t = F.pad(QE_t, (0, T-1)).view(B, H, T+1, 2*T-1)
    return QE_t[:, :, :T, -T:]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_head, n_head, dropout=0.1, pre_layernorm=False):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_head = n_head
        self.pre_layernorm = pre_layernorm

        self.W_q = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_out = nn.Linear(n_head * d_head, d_model)

        self.atten_scale = 1 / (d_head ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    
    def forward(self, x, context, mask=None, lookup_table=None):
        B, T1, D = x.shape
        _, T2, _ = context.shape

        if self.pre_layernorm:
            x = self.layer_norm(x)

        # linear projection
        q = self.W_q(x) # (B, T1, n_head*d_head)
        k = self.W_k(context) # (B, T2, n_head*d_head)
        v = self.W_v(context) # (B, T2, n_head*d_head)

        # split heads
        q = q.view(B, T1, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T1, d_head)
        k = k.view(B, T2, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T2, d_head)
        v = v.view(B, T2, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T2, d_head)

        # attention score
        atten_score = torch.matmul(q, k.transpose(-2, -1)) # (B, n_head, T1, T2)

        # relative positional encoding
        if lookup_table is not None:
            atten_score += skew(torch.matmul(q, lookup_table.transpose(-2, -1)))
        
        # attention scale
        atten_score *= self.atten_scale # (B, n_head, T1, T2)

        # mask
        if mask is not None:
            atten_score.masked_fill_(mask, -1e9)
        
        # attention
        attention = F.softmax(atten_score, dim=-1) # (B, n_head, T1, T2)
        attention = torch.matmul(attention, v).transpose(1, 2).contiguous().view(B, -1, self.n_head * self.d_head) # (B, T1, n_head*d_head)

        # output
        output = self.W_out(attention) # (B, T1, d_model)
        output = self.dropout(output)

        if self.pre_layernorm:
            return x + output
        else:
            return self.layer_norm(x + output)

class LocalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_head, n_head, receptive_size, dropout=0.1, pre_layernorm=False):
        super(LocalMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_head = n_head
        self.receptive_size = receptive_size
        if receptive_size % 2 == 0:
            raise ValueError("receptive size must be odd")
        self.pre_layernorm = pre_layernorm

        self.W_q = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_out = nn.Linear(n_head * d_head, d_model)

        self.atten_scale = 1 / (d_head ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def local_attention_mask(self, Q_Kt):
        B, H, T1, T2 = Q_Kt.shape

        mask = torch.ones((B, H, T1, T2+1), dtype=torch.bool, device=Q_Kt.device)
        mask = F.pad(mask, (self.receptive_size, 0), value=False).reshape(B, H, T1*(T2+self.receptive_size+1))
        mask = mask[..., :-T1].reshape(B, H, T1, T2+self.receptive_size)
        mask = mask[:, :, :, self.receptive_size//2:T2+self.receptive_size//2]
        
        return mask

    def forward(self, x, context, lookup_table=None):
        B, T1, D = x.shape
        _, T2, _ = context.shape

        if self.pre_layernorm:
            x = self.layer_norm(x)

        # linear projection
        q = self.W_q(x) # (B, T1, n_head*d_head)
        k = self.W_k(context) # (B, T2, n_head*d_head)
        v = self.W_v(context) # (B, T2, n_head*d_head)

        # split heads
        q = q.view(B, T1, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T1, d_head)
        k = k.view(B, T2, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T2, d_head)
        v = v.view(B, T2, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T2, d_head)

        # attention score
        atten_score = torch.matmul(q, k.transpose(-2, -1)) # (B, n_head, T1, T2)

        # relative positional encoding
        if lookup_table is not None:
            atten_score += skew(torch.matmul(q, lookup_table.transpose(-2, -1)))
        
        # attention scale
        atten_score *= self.atten_scale # (B, n_head, T1, T2)
        
        # local attention
        atten_mask = self.local_attention_mask(atten_score)
        atten_score.masked_fill_(atten_mask, -1e9)
        
        # attention
        attention = F.softmax(atten_score, dim=-1) # (B, n_head, T1, T2)
        attention = torch.matmul(attention, v).transpose(1, 2).contiguous().view(B, -1, self.n_head * self.d_head) # (B, T1, n_head*d_head)

        # output
        output = self.W_out(attention) # (B, T1, d_model)
        output = self.dropout(output)

        if self.pre_layernorm:
            return x + output
        else:
            return self.layer_norm(x + output)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, pre_layernorm=False):
        super(PoswiseFeedForwardNet, self).__init__()
        self.pre_layernorm = pre_layernorm

        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        if self.pre_layernorm:
            return x + self.layers(self.layer_norm(x))
        else:
            return self.layer_norm(x + self.layers(x))