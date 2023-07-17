import numpy as np

from . import rotmat, quat, aaxis

"""
6D rotation to other representation
"""
def to_quat(rot6d):
    return rotmat.to_quat(to_rotmat(rot6d))

def to_rotmat(rot6d):
    x_, y_ = rot6d[..., :3], rot6d[..., 3:]

    # normalize x
    x = x_ / np.linalg.norm(x_, axis=-1, keepdims=True)

    # normalize y
    y = y_ - np.sum(x * y_, axis=-1, keepdims=True) * x
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)

    # normalize z
    z = np.cross(x, y, axis=-1)

    return np.stack([x, y, z], axis=-2) # (..., 3, 3)

def to_aaxis(rot6d):
    return rotmat.to_aaxis(to_rotmat(rot6d))

def to_xform(rot6d, translation=None):
    return rotmat.to_xform(to_rotmat(rot6d), translation=translation)

"""
Other representation to 6D rotation
"""
def from_aaxis(a):
    return aaxis.to_rot6d(a)

def from_quat(q):
    return quat.to_rot6d(q)

def from_rotmat(r):
    return rotmat.to_rot6d(r)

def from_xform(x):
    return rotmat.to_rot6d(x)