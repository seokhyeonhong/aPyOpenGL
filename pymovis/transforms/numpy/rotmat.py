import numpy as np

from . import quat, aaxis, rot6d, xform

"""
Rotation matrix to other representation
"""
def to_aaxis(rotmat):
    return quat.to_aaxis(to_quat(rotmat))

def to_quat(rotmat):
    batch_dim = rotmat.shape[:-2]
    rotmat_ = rotmat.reshape(batch_dim + (9,))
    rotmat00, rotmat01, rotmat02, rotmat10, rotmat11, rotmat12, rotmat20, rotmat21, rotmat22 = rotmat_[..., 0], rotmat_[..., 1], rotmat_[..., 2], rotmat_[..., 3], rotmat_[..., 4], rotmat_[..., 5], rotmat_[..., 6], rotmat_[..., 7], rotmat_[..., 8]

    def _to_positive_sqrt(x):
        ret = np.zeros_like(x)
        positive = x > 0
        ret[positive] = np.sqrt(x[positive])
        return ret

    quat_square = np.stack([
        (1.0 + rotmat00 + rotmat11 + rotmat22), # 4*r*r
        (1.0 + rotmat00 - rotmat11 - rotmat22), # 4*i*i
        (1.0 - rotmat00 + rotmat11 - rotmat22), # 4*j*j
        (1.0 - rotmat00 - rotmat11 + rotmat22), # 4*k*k
    ], axis=-1) # (..., 4)
    quat_abs = _to_positive_sqrt(quat_square) # 2*|r|, 2*|i|, 2*|j|, 2*|k|
    r, i, j, k = quat_abs[..., 0], quat_abs[..., 1], quat_abs[..., 2], quat_abs[..., 3]

    quat_candidates = np.stack([
        np.stack([r*r, rotmat21-rotmat12, rotmat02-rotmat20, rotmat10-rotmat01], axis=-1),
        np.stack([rotmat21-rotmat12, i*i, rotmat01+rotmat10, rotmat02+rotmat20], axis=-1),
        np.stack([rotmat02-rotmat20, rotmat01+rotmat10, j*j, rotmat12+rotmat21], axis=-1),
        np.stack([rotmat10-rotmat01, rotmat02+rotmat20, rotmat12+rotmat21, k*k], axis=-1),
    ], axis=-2) # (..., 4, 4)
    quat_candidates = quat_candidates / (2 * quat_abs[..., None] + 1e-8)

    quat_idx = np.argmax(quat_square, axis=-1)
    quat = np.take_along_axis(quat_candidates, quat_idx[..., None, None].repeat(4, axis=-1), axis=-2).squeeze(-2)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)

    return quat.reshape(batch_dim + (4,))

def to_rot6d(rotmat):
    return np.concatenate([rotmat[..., 0, :], rotmat[..., 1, :]], axis=-1)

def to_xform(rotmat, translation=None):
    batch_dims = rotmat.shape[:-2]

    # transformation matrix
    I = np.eye(4, dtype=np.float32) # (4, 4)
    I = np.tile(I, reps=batch_dims + (1, 1)) # (..., 4, 4)

    # fill rotation matrix
    I[..., :3, :3] = rotmat # (..., 4, 4)

    # fill translation
    if translation is not None:
        I[..., :3, 3] = translation

    return I

"""
Other representation to rotation matrix
"""
def from_aaxis(a):
    return aaxis.to_rotmat(a)

def from_quat(q):
    return quat.to_rotmat(q)

def from_rot6d(r):
    return rot6d.to_rotmat(r)

def from_xform(x):
    return xform.to_rotmat(x)