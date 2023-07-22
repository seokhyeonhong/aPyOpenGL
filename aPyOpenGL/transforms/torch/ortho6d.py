import torch
import torch.nn.functional as F

from . import rotmat, quat, aaxis, xform

"""
Operations
"""
def fk(local_ortho6d, root_pos, skeleton):
    global_xforms = xform.fk(to_xform(local_ortho6d), root_pos, skeleton)
    global_ortho6ds, global_pos = xform.to_ortho6d(global_xforms), xform.to_translation(global_xforms)
    return global_ortho6ds, global_pos

"""
Orthonormal 6D rotation to other representation
"""
def to_rotmat(ortho6d):
    x_, y_ = ortho6d[..., :3], ortho6d[..., 3:]
    x = F.normalize(x_, dim=-1, eps=1e-8)
    y = F.normalize(y_ - (x * y_).sum(dim=-1, keepdim=True) * x, dim=-1, eps=1e-8)
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], dim=-2) # (..., 3, 3)

def to_aaxis(ortho6d):
    return rotmat.to_aaxis(to_rotmat(ortho6d))

def to_quat(ortho6d):
    return rotmat.to_quat(to_rotmat(ortho6d))

def to_xform(ortho6d, translation=None):
    return rotmat.to_xform(to_rotmat(ortho6d), translation=translation)

"""
Other representation to 6D rotation
"""
def from_aaxis(a):
    return aaxis.to_ortho6d(a)

def from_quat(q):
    return quat.to_ortho6d(q)

def from_rotmat(r):
    return rotmat.to_ortho6d(r)

def from_xform(x):
    return xform.to_ortho6d(x)