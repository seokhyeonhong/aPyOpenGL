import torch
import torch.nn.functional as F

from . import rotmat

def to_rotmat(rot6d):
    x_, y_ = rot6d[..., :3], rot6d[..., 3:]
    x = F.normalize(x_, dim=-1)
    y = F.normalize(y_ - (x * y_).sum(dim=-1, keepdim=True) * x, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], dim=-2) # (..., 3, 3)

def to_quat(rot6d):
    return rotmat.to_quat(to_rotmat(rot6d))

def to_aaxis(rot6d):
    return rotmat.to_aaxis(to_rotmat(rot6d))