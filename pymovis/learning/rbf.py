import torch
import numpy as np
import scipy.linalg as linalg
import scipy.spatial as spatial

KERNELS = {
    "multiquadric": lambda x: torch.sqrt(x**2 + 1),
    "inverse":      lambda x: 1.0 / torch.sqrt(x**2 + 1),
    "gaussian":     lambda x: torch.exp(-x**2),
    "linear":       lambda x: x,
    "quadric":      lambda x: x**2,
    "cubic":        lambda x: x**3,
    "quartic":      lambda x: x**4,
    "quintic":      lambda x: x**5,
    "thin_plate":   lambda x: x**2 * torch.log(x + 1e-8),
    "logistic":     lambda x: 1.0 / (1.0 + torch.exp(-torch.clamp(x, -5, 5))),
    "smoothstep":   lambda x: ((torch.clamp(1.0 - x, 0.0, 1.0))**2.0) * (3 - 2*(torch.clamp(1.0 - x, 0.0, 1.0)))
}

class Solve:
    def __init__(self, l=1e-5):
        self.l = l
        
    def fit(self, X, Y):
        """
        Args:
            X: (N, N) or (B, N, N)
            Y: (N, K) or (B, N, K)
        """
        LU, piv = torch.linalg.lu_factor(X.transpose(-1, -2) + torch.eye(X.shape[-2]) * self.l)
        self.M = torch.lu_solve(Y, LU, piv).transpose(-1, -2)
        return LU, piv
        
    def forward(self, Xp):
        """
        Args:
            Xp: (M, N) or (B, M, N)
        Returns:
            (M, K) or (B, M, K)
        """
        return self.M.matmul(Xp.transpose(-1, -2)).transpose(-1, -2)

class RBF:
    def __init__(self, L=None, eps=None, function="multiquadric", smooth=1e-8):
        self.solver = Solve(l=-smooth) if L is None else L
        self.kernel = KERNELS.get(function)
        if self.kernel is None:
            raise ValueError(f"Invalid kernel function: {function}")
        self.eps = eps
        
    def fit(self, X, Y):
        """
        Args:
            X: (B, N, D) or (N, D)
            Y: (B, N, K) or (N, K)
        """
        self.X = X
        dist = torch.cdist(self.X, self.X) # (B, N, N) or (N, N)
        self.eps = torch.ones(len(dist)) / dist.mean() if self.eps is None else self.eps
        return self.solver.fit(self.kernel(self.eps * dist), Y)
        
    def forward(self, Xp):
        """
        Args:
            Xp: (B, M, D) or (M, D)
        Returns:
            (B, M, K) or (M, K)
        """
        D = torch.cdist(Xp, self.X)
        return self.solver.forward(self.kernel(self.eps * D))