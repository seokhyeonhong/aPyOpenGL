import torch

from . import rotmat, quat, ortho6d, xform

def _split_axis_angle(aaxis):
    angle = torch.norm(aaxis, dim=-1)
    axis = aaxis / (angle[..., None] + 1e-8)
    return angle, axis

"""
Angle-axis to other representations
"""
def to_quat(aaxis):
    angle, axis = _split_axis_angle(aaxis)

    cos = torch.cos(angle / 2)[..., None]
    sin = torch.sin(angle / 2)[..., None]
    axis_sin = axis * sin

    return torch.cat([cos, axis_sin], dim=-1)

def to_rotmat(aaxis):
    # split angle and axis
    angle, axis = _split_axis_angle(aaxis)
    a0, a1, a2 = axis[..., 0], axis[..., 1], axis[..., 2]
    zero       = torch.zeros_like(a0)
    batch_dims = angle.shape

    # skew symmetric matrix
    S   = torch.stack([zero, -a2, a1, a2, zero, -a0, -a1, a0, zero], dim=-1)
    S   = S.reshape(batch_dims + (3, 3))

    # rotation matrix
    I   = torch.eye(3, dtype=torch.float32)
    I   = I.repeat(batch_dims + (1, 1))
    sin = torch.sin(angle)[..., None, None]
    cos = torch.cos(angle)[..., None, None]

    return I + S * sin + torch.matmul(S, S) * (1 - cos)

def to_ortho6d(aaxis):
    return rotmat.to_ortho6d(to_rotmat(aaxis))

def to_xform(aaxis, translation=None):
    return rotmat.to_xform(to_rotmat(aaxis), translation=translation)

"""
Other representations to angle-axis
"""
def from_quat(q):
    return quat.to_aaxis(q)

def from_rotmat(r):
    return rotmat.to_aaxis(r)

def from_ortho6d(r):
    return ortho6d.to_aaxis(r)

def from_xform(x):
    return xform.to_aaxis(x)