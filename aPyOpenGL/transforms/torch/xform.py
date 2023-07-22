import torch

from . import rotmat, quat, aaxis

"""
Operations
"""
def interpolate(x_from, x_to, t):
    r_from, p_from = x_from[..., :3, :3], x_from[..., :3, 3]
    r_to,   p_to   = x_to[..., :3, :3], x_to[..., :3, 3]

    r = rotmat.interpolate(r_from, r_to, t)
    p = p_from + (p_to - p_from) * t

    return rotmat.to_xform(r, translation=p)

"""
Transformation matrix to other representation
"""
def to_rotmat(xform):
    return xform[..., :3, :3]

def to_quat(xform):
    return rotmat.to_quat(to_rotmat(xform))

def to_aaxis(xform):
    return quat.to_aaxis(to_quat(xform))

def to_ortho6d(xform):
    return rotmat.to_ortho6d(to_rotmat(xform))

def to_trans(xform):
    return xform[..., :3, 3]

"""
Other representation to transformation matrix
"""
def from_rotmat(r, translation=None):
    return rotmat.to_xform(r, translation=translation)

def from_quat(q, translation=None):
    return quat.to_xform(q, translation=translation)

def from_aaxis(a, translation=None):
    return aaxis.to_xform(a, translation=translation)

def from_ortho6d(r, translation=None):
    return rotmat.to_xform(r, translation=translation)