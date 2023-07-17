import numpy as np

from . import rotmat, quat, aaxis

"""
Operations
"""
def interpolate(x0, x1, t):
    r0, p0 = x0[..., :3, :3], x0[..., :3, 3]
    r1, p1 = x1[..., :3, :3], x1[..., :3, 3]

    r = rotmat.interpolate(r0, r1, t)
    p = p0 + (p1 - p0) * t

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

def to_rot6d(xform):
    return rotmat.to_rot6d(to_rotmat(xform))

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

def from_rot6d(r, translation=None):
    return rotmat.to_xform(r, translation=translation)