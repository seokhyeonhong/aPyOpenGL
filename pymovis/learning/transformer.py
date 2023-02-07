import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_head, n_head, dropout=0.1, pre_layernorm=True):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_head = n_head
        self.pre_layernorm = pre_layernorm

        self.W_q = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_o = nn.Linear(n_head * d_head, d_model)

        self.atten_scale = 1 / (d_head ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        B, T, D = x.shape

        if self.pre_layernorm:
            x = self.layer_norm(x)

        # linear projection to (B, T, n_head*d_head)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # split heads to (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # attention score (B, n_head, T, T)
        atten_score = torch.matmul(q, k.transpose(-2, -1)) * self.atten_scale
        if mask is not None:
            atten_score.masked_fill_(mask, -1e9)
        
        # attention (B, T, n_head*d_head)
        attention = F.softmax(atten_score, dim=-1)
        attention = torch.matmul(attention, v).transpose(1, 2).contiguous().view(B, -1, self.n_head * self.d_head)

        # output (B, T, D)
        attention = self.W_o(attention)
        attention = self.dropout(attention)

        return x + attention if self.pre_layernorm else self.layer_norm(x + attention)

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_head, n_head, dropout=0.1, pre_layernorm=True):
        super(RelativeMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_head = n_head
        self.pre_layernorm = pre_layernorm

        self.W_q = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_o = nn.Linear(n_head * d_head, d_model)

        self.atten_scale = 1 / (d_head ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def skew(self, QE_t):
        B, H, T, _ = QE_t.shape # (B, H, T, 2T-1)
        QE_t = F.pad(QE_t, (0, 1)).view(B, H, -1)
        QE_t = F.pad(QE_t, (0, T - 1)).view(B, H, T+1, 2*T - 1)
        return QE_t[:, :, :T, -T:]

    def forward(self, x, lookup_table, mask=None):
        B, T, D = x.shape

        if self.pre_layernorm:
            x = self.layer_norm(x)

        # linear projection to (B, T, n_head*d_head)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # split heads to (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # attention score (B, n_head, T, T)
        atten_score = torch.matmul(q, k.transpose(-2, -1))
        rel_atten_score = self.skew(torch.matmul(q, lookup_table.transpose(-2, -1)))
        atten_score = (atten_score + rel_atten_score) * self.atten_scale

        if mask is not None:
            atten_score.masked_fill_(mask, -1e9)

        # attention (B, T, n_head*d_head)
        attention = F.softmax(atten_score, dim=-1)
        attention = torch.matmul(attention, v).transpose(1, 2).contiguous().view(B, -1, self.n_head * self.d_head)

        # output (B, T, D)
        attention = self.W_o(attention)
        attention = self.dropout(attention)
        
        if self.pre_layernorm:
            return x + attention
        else:
            return self.layer_norm(x + attention)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, pre_layernorm=True):
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