import torch

"""
Functions for constant tensors
"""

def X(device="cpu"):
    return torch.tensor([1, 0, 0], dtype=torch.float32, device=device)

def Y(device="cpu"):
    return torch.tensor([0, 1, 0], dtype=torch.float32, device=device)

def Z(device="cpu"):
    return torch.tensor([0, 0, 1], dtype=torch.float32, device=device)

def XY(device="cpu"):
    return torch.tensor([1, 1, 0], dtype=torch.float32, device=device)

def XZ(device="cpu"):
    return torch.tensor([1, 0, 1], dtype=torch.float32, device=device)

def YZ(device="cpu"):
    return torch.tensor([0, 1, 1], dtype=torch.float32, device=device)

def FORWARD(device="cpu"):
    return torch.tensor([0, 0, 1], dtype=torch.float32, device=device)

def UP(device="cpu"):
    return torch.tensor([0, 1, 0], dtype=torch.float32, device=device)

def LEFT(device="cpu"):
    return torch.tensor([1, 0, 0], dtype=torch.float32, device=device)

def P_ZERO(device="cpu"):
    return torch.zeros(3, dtype=torch.float32, device=device)

def EPSILON():
    return torch.finfo(torch.float32).eps

def INFINITY():
    return torch.finfo(torch.float32).max