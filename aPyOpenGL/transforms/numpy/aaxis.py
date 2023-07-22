import numpy as np

from . import rotmat, quat, ortho6d, xform

def _split_axis_angle(aaxis):
    # aaxis: (..., 3) angle axis
    angle = np.linalg.norm(aaxis, axis=-1)
    axis = aaxis / (angle[..., None] + 1e-8)
    return angle, axis

"""
Angle-axis to other representations
"""
def to_quat(aaxis):
    angle, axis = _split_axis_angle(aaxis)

    cos = np.cos(angle / 2)[..., None]
    sin = np.sin(angle / 2)[..., None]
    axis_sin = axis * sin

    return np.concatenate([cos, axis_sin], axis=-1) # (..., 4)

def to_rotmat(aaxis):
    # split angle and axis
    angle, axis = _split_axis_angle(aaxis)
    a0, a1, a2 = axis[..., 0], axis[..., 1], axis[..., 2]
    zero       = np.zeros_like(a0)
    batch_dims = angle.shape

    # skew symmetric matrix
    S   = np.stack([zero, -a2, a1, a2, zero, -a0, -a1, a0, zero], axis=-1)
    S   = S.reshape(batch_dims + (3, 3))             # (..., 3, 3)

    # rotation matrix
    I   = np.eye(3, dtype=np.float32)                 # (3, 3)
    I   = np.tile(I, reps=(batch_dims + (1, 1)))      # (..., 3, 3)
    sin = np.sin(angle)[..., None, None]              # (..., 1, 1)
    cos = np.cos(angle)[..., None, None]              # (..., 1, 1)

    return I + S * sin + np.matmul(S, S) * (1 - cos)  # (..., 3, 3)

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