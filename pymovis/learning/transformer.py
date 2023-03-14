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
        self.W_out = nn.Linear(n_head * d_head, d_model)

        self.atten_scale = 1 / (d_head ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, context, mask=None):
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
        atten_score = torch.matmul(q, k.transpose(-2, -1)) * self.atten_scale # (B, n_head, T1, T2)
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
        self.W_out = nn.Linear(n_head * d_head, d_model)

        self.atten_scale = 1 / (d_head ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def skew(self, QE_t):
        B, H, T, X = QE_t.shape # (B, H, T, 2T-1)

        QE_t = F.pad(QE_t, (0, 1)).view(B, H, 2*T*T)
        QE_t = F.pad(QE_t, (0, T-1)).view(B, H, T+1, 2*T - 1)
        return QE_t[:, :, :T, -T:]

    def forward(self, x, context, lookup_table, mask=None):
        B, T1, D = x.shape
        _, T2, _ = context.shape

        if self.pre_layernorm:
            x = self.layer_norm(x)

        # linear projection to
        q = self.W_q(x) # (B, T1, n_head*d_head)
        k = self.W_k(context) # (B, T2, n_head*d_head)
        v = self.W_v(context) # (B, T2, n_head*d_head)

        # split heads
        q = q.view(B, T1, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T1, d_head)
        k = k.view(B, T2, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T2, d_head)
        v = v.view(B, T2, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T2, d_head)

        # attention score
        atten_score = torch.matmul(q, k.transpose(-2, -1)) # (B, n_head, T1, T2)
        rel_atten_score = self.skew(torch.matmul(q, lookup_table.transpose(-2, -1))) # (B, n_head, T1, T1)
        atten_score = (atten_score + rel_atten_score) * self.atten_scale # TODO: Fix this line for atten_score and rel_atten_score are not the same shape

        if mask is not None:
            atten_score.masked_fill_(mask, -1e9)

        # attention
        attention = F.softmax(atten_score, dim=-1)
        attention = torch.matmul(attention, v).transpose(1, 2).contiguous().view(B, -1, self.n_head * self.d_head) # (B, T1, n_head*d_head)

        # output
        output = self.W_out(attention) # (B, T1, d_model)
        output = self.dropout(output)
        
        if self.pre_layernorm:
            return x + output
        else:
            return self.layer_norm(x + output)

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