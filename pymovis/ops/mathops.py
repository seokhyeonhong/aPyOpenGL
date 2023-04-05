import torch
import torch.nn.functional as F
import numpy as np

####################################################################################

def normalize_torch(x, dim=-1):
    return F.normalize(x, dim=dim)

def normalize_numpy(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)

def normalize(x, dim=-1):
    """
    Normalize a tensor or numpy array along a given dimension
    Args:
        x: torch.Tensor or numpy.ndarray
        dim: dimension to normalize along
    Returns:
        normalized tensor or array
    """
    if isinstance(x, torch.Tensor):
        return normalize_torch(x, dim)
    elif isinstance(x, np.ndarray):
        return normalize_numpy(x, dim)
    else:
        raise TypeError(f"Type must be torch.Tensor or numpy.ndarray, but got {type(x)}")

####################################################################################

def signed_angle_torch(v1, v2, vn, dim=-1, eps=1e-6):
    v1_unit = F.normalize(v1, dim=dim)
    v2_unit = F.normalize(v2, dim=dim)

    dot = torch.sum(v1_unit * v2_unit, dim=dim)
    dot = torch.clamp(dot, -1 + eps, 1 - eps)
    angle = torch.acos(dot)

    cross = torch.cross(v1_unit, v2_unit, dim=dim)
    cross = torch.sum(cross * vn, dim=dim)
    angle = torch.where(cross < 0, -angle, angle)

    return angle

def signed_angle_numpy(v1, v2, vn, dim=-1):
    v1_unit = v1 / (np.linalg.norm(v1, axis=dim, keepdims=True) + 1e-8)
    v2_unit = v2 / (np.linalg.norm(v2, axis=dim, keepdims=True) + 1e-8)

    dot = np.sum(v1_unit * v2_unit, axis=dim)
    dot = np.clip(dot, -1, 1)
    angle = np.arccos(dot)

    cross = np.cross(v1_unit, v2_unit, axis=dim)
    cross = np.sum(cross * vn, axis=dim)
    angle = np.where(cross < 0, -angle, angle)

    return angle

def signed_angle(v1, v2, vn=[0, 1, 0], dim=-1):
    """
    Signed angle from v1 to v2 around vn
    Args:
        v1: vector to rotate from (..., 3)
        v2: vector to rotate to (..., 3)
        vn: normal vector to rotate around (..., 3)
        dim: dimension to normalize along
    Returns:
        Signed angle from v1 to v2 around vn (...,)
    """
    if isinstance(v1, torch.Tensor):
        return signed_angle_torch(v1, v2, torch.tensor(vn, dtype=torch.float32, device=v1.device), dim)
    elif isinstance(v1, np.ndarray):
        return signed_angle_numpy(v1, v2, np.array(vn, dtype=np.float32), dim)
    else:
        raise TypeError(f"Type must be torch.Tensor or numpy.ndarray, but got {type(v1)}")

####################################################################################

def lerp(x, y, t):
    """
    Args:
        x: start value (..., D)
        y: end value (..., D)
        t: interpolation value (...,) or float
    Returns:
        interpolated value (..., D)
    """
    return x + t * (y - x)

####################################################################################

def clamp(x, min_val, max_val):
    if isinstance(x, torch.Tensor):
        return torch.clamp(x, min_val, max_val)
    elif isinstance(x, np.ndarray):
        return np.clip(x, min_val, max_val)
    else:
        return max(min(x, max_val), min_val)