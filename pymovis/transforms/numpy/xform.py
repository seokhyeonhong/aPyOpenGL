import numpy as np

from . import rotmat, quat, aaxis

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