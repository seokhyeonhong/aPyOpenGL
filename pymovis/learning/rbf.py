import torch
import torch.nn as nn

def multiquadric(x):
    return torch.sqrt(x**2 + 1)

def inverse(x):
    return 1.0 / torch.sqrt(x**2 + 1)

def gaussian(x):
    return torch.exp(-x**2)

def linear(x):
    return x

def quadric(x):
    return x**2

def cubic(x):
    return x**3

def quartic(x):
    return x**4

def quintic(x):
    return x**5

def thin_plate(x):
    return x**2 * torch.log(x + 1e-8)

def logistic(x):
    return 1.0 / (1.0 + torch.exp(-torch.clamp(x, -5, 5)))

def smoothstep(x):
    return ((torch.clamp(1.0 - x, 0.0, 1.0))**2.0) * (3 - 2*(torch.clamp(1.0 - x, 0.0, 1.0)))

KERNELS = {
    "multiquadric": multiquadric,
    "inverse":      inverse,
    "gaussian":     gaussian,
    "linear":       linear,
    "quadric":      quadric,
    "cubic":        cubic,
    "quartic":      quartic,
    "quintic":      quintic,
    "thin_plate":   thin_plate,
    "logistic":     logistic,
    "smoothstep":   smoothstep,
}


class Solve(nn.Module):
    def __init__(self, l=1e-5):
        super(Solve, self).__init__()
        self.l = l
        
    def fit(self, X, Y):
        """
        Args:
            X: (..., N, N)
            Y: (..., N, K)
        """
        LU, piv = torch.linalg.lu_factor(X.transpose(-1, -2) + torch.eye(X.shape[-2], device=X.device) * self.l)
        self.M = nn.Parameter(torch.lu_solve(Y, LU, piv).transpose(-1, -2))
        
    def forward(self, Xp):
        """
        Args:
            Xp: (..., M, N)
        Returns:
            (..., M, K)
        """
        return torch.matmul(self.M, Xp.transpose(-1, -2)).transpose(-1, -2)

class RBF(nn.Module):
    def __init__(self, L=None, eps=None, function="multiquadric", smooth=1e-8):
        super(RBF, self).__init__()
        self.solver = Solve(l=-smooth) if L is None else L
        self.kernel = KERNELS.get(function)
        if self.kernel is None:
            raise ValueError(f"Invalid kernel function: {function}")
        self.eps = eps
        
    def fit(self, X, Y):
        """
        Args:
            X: (..., N, D)
            Y: (..., N, K)
        """
        self.X = X
        dist = torch.cdist(self.X, self.X) # (B, N, N) or (N, N)
        self.eps = torch.ones(len(dist), device=X.device) / dist.mean() if self.eps is None else self.eps
        self.solver.fit(self.kernel(self.eps * dist), Y)
        
    def forward(self, Xp):
        """
        Args:
            Xp: (..., M, D)
        Returns:
            (..., M, K)
        """
        D = torch.cdist(Xp, self.X)
        return self.solver.forward(self.kernel(self.eps * D))