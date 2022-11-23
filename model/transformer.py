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
    
    def forward(self, x, E_rel=None, mask=None):
        B = x.shape[0]

        Q = self.W_q(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # compute attention
        attention = torch.matmul(Q, K.transpose(-2, -1))
        if E_rel is not None:
            S_rel = skew_relative(torch.matmul(Q, E_rel.transpose(-2, -1)))
            attention += S_rel

        attention = attention * self.scaling_factor

        if mask is not None:
            attention = attention + mask

        attention = F.softmax(attention, dim=-1)
        attention = torch.matmul(attention, V).transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.head_dim)
        attention = self.W_o(attention)
        attention = self.dropout(attention)
        return attention

class PhaseMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, head_dim, output_dim, num_heads, dropout=0.1):
        super(PhaseMultiHeadAttention, self).__init__()
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
    
    def forward(self, x, phase, E_rel=None, mask=None):
        num_batch = x.shape[0]

        Q = self.W_q(x).view(num_batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(num_batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(num_batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # compute attention
        attention = torch.matmul(Q, K.transpose(-2, -1))
        if E_rel != None:
            S_rel = skew_relative(torch.matmul(Q, E_rel.transpose(-2, -1)))
            attention += S_rel

        attention = attention * self.scaling_factor

        if mask != None:
            attention = attention + mask

        attention = F.softmax(attention, dim=-1)
        attention = torch.matmul(attention, V)
        attention = (attention * phase).transpose(1, 2).contiguous().view(num_batch, -1, self.num_heads * self.head_dim)
        attention = self.W_o(attention)
        attention = self.dropout(attention)
        return attention