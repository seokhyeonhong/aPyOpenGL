import numpy as np
import scipy.linalg as linalg
import scipy.spatial as spatial

KERNELS = {
    "multiquadric": lambda x: np.sqrt(x**2 + 1),
    "inverse":      lambda x: 1.0 / np.sqrt(x**2 + 1),
    "gaussian":     lambda x: np.exp(-x**2),
    "linear":       lambda x: x,
    "quadric":      lambda x: x**2,
    "cubic":        lambda x: x**3,
    "quartic":      lambda x: x**4,
    "quintic":      lambda x: x**5,
    "thin_plate":   lambda x: x**2 * np.log(x + 1e-8),
    "logistic":     lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -5, 5))),
    "smoothstep":   lambda x: ((np.clip(1.0 - x, 0.0, 1.0))**2.0) * (3 - 2*(np.clip(1.0 - x, 0.0, 1.0)))
}

class Solve:
    def __init__(self, l=1e-5):
        self.l = l
        
    def fit(self, X, Y):
        self.M = linalg.lu_solve(linalg.lu_factor(X.T + np.eye(len(X)) * self.l), Y).T
        
    def __call__(self, Xp):
        return self.M.dot(Xp.T).T
    
class RBF:
    def __init__(self, L=None, eps=None, function="multiquadric", smooth=1e-8):
        self.solver = Solve(l=-smooth) if L is None else L
        self.kernel = KERNELS.get(function)
        if self.kernel is None:
            raise ValueError(f"Invalid kernel function: {function}")
        self.eps = eps
        
    def fit(self, X, Y):
        self.X = X
        dist = spatial.distance.cdist(self.X, self.X)
        self.eps = np.ones(len(dist)) / dist.mean() if self.eps is None else self.eps
        self.solver.fit(self.kernel(self.eps * dist), Y)
        
    def forward(self, Xp):
        D = spatial.distance.cdist(Xp, self.X)
        return self.solver(self.kernel(self.eps * D))