import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len):
        super(PositionalEmbedding, self).__init__()

        if dim % 2 != 0:
            raise ValueError(f"PositionalEmbedding: dim must be even, but got {dim}")

        pos = torch.arange(0, max_len, step=1, dtype=torch.float32).unsqueeze(1)
        div_term = 1.0 / torch.pow(10000, torch.arange(0, dim, step=2, dtype=torch.float32) / dim)

        embedding = torch.empty((max_len, dim))
        embedding[:, 0::2] = torch.sin(pos * div_term)
        embedding[:, 1::2] = torch.cos(pos * div_term)
        self.embedding = nn.Parameter(embedding, requires_grad=False)
        
    def forward(self, position):
        return self.embedding[min(position, self.embedding.shape[0] - 1)]