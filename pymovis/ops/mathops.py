import torch
import torch.nn.functional as F
import numpy as np

####################################################################################

def normalize_torch(x, dim=-1):
    return F.normalize(x, dim=dim)

def normalize_numpy(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)

def normalize(x, dim=-1):
    if isinstance(x, torch.Tensor):
        return normalize_torch(x, dim)
    elif isinstance(x, np.ndarray):
        return normalize_numpy(x, dim)
    else:
        raise TypeError(f"Type must be torch.Tensor or numpy.ndarray, but got {type(x)}")

####################################################################################

def signed_angle_torch(v1, v2, vn=torch.tensor([0, 1, 0], dtype=torch.float32), dim=-1):
    v1_unit = F.normalize(v1, dim=dim)
    v2_unit = F.normalize(v2, dim=dim)

    dot = torch.sum(v1_unit * v2_unit, dim=dim)
    dot = torch.clamp(dot, -1, 1)
    angle = torch.acos(dot)

    cross = torch.cross(v1_unit, v2_unit, dim=dim)
    cross = torch.sum(cross * vn, dim=dim)
    angle = torch.where(cross < 0, -angle, angle)

    return angle

def signed_angle_numpy(v1, v2, vn=np.array([0, 1, 0], dtype=np.float32), dim=-1):
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
    """ Signed angle from v1 to v2 around vn """
    if isinstance(v1, torch.Tensor):
        return signed_angle_torch(v1, v2, torch.tensor(vn, dtype=torch.float32), dim)
    elif isinstance(v1, np.ndarray):
        return signed_angle_numpy(v1, v2, np.array(vn, dtype=np.float32), dim)
    else:
        raise TypeError(f"Type must be torch.Tensor or numpy.ndarray, but got {type(v1)}")