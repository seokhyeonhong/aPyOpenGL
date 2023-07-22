import torch
import torch.nn.functional as F

from . import quat, aaxis, ortho6d, xform

"""
Operations
"""
def interpolate(r_from, r_to, t):
    q_from = to_quat(r_from)
    q_to   = to_quat(r_to)
    q = quat.interpolate(q_from, q_to, t)
    return quat.to_rotmat(q)

"""
Rotations to other representations
"""
def to_aaxis(rotmat):
    return quat.to_aaxis(to_quat(rotmat))

def to_quat(rotmat):
    batch_dim = rotmat.shape[:-2]
    R00, R01, R02, R10, R11, R12, R20, R21, R22 = torch.unbind(rotmat.reshape(batch_dim + (9,)), dim=-1)

    def _to_positive_sqrt(x):
        ret = torch.zeros_like(x)
        positive = x > 0
        ret[positive] = torch.sqrt(x[positive])
        return ret

    Q_square = torch.stack([
        (1.0 + R00 + R11 + R22), # 4*r*r
        (1.0 + R00 - R11 - R22), # 4*i*i
        (1.0 - R00 + R11 - R22), # 4*j*j
        (1.0 - R00 - R11 + R22), # 4*k*k
    ], dim=-1) # (..., 4)
    Q_abs = _to_positive_sqrt(Q_square) # 2*|r|, 2*|i|, 2*|j|, 2*|k|
    r, i, j, k = torch.unbind(Q_abs, dim=-1)

    Q_candidates = torch.stack([
        torch.stack([r*r, R21-R12, R02-R20, R10-R01], dim=-1),
        torch.stack([R21-R12, i*i, R01+R10, R02+R20], dim=-1),
        torch.stack([R02-R20, R01+R10, j*j, R12+R21], dim=-1),
        torch.stack([R10-R01, R02+R20, R12+R21, k*k], dim=-1),
    ], dim=-2) # (..., 4, 4)
    Q_candidates = Q_candidates / (2 * Q_abs[..., None] + 1e-8)

    Q_idx = torch.argmax(Q_square, dim=-1)
    Q = torch.gather(Q_candidates, dim=-2, index=Q_idx[..., None, None].expand(batch_dim + (1, 4))).squeeze(-2)
    Q = F.normalize(Q, dim=-1)
    
    return Q.reshape(batch_dim + (4,))

def to_ortho6d(rotmat):
    return torch.cat([rotmat[..., 0, :], rotmat[..., 1, :]], dim=-1)

def to_xform(rotmat, translation=None):
    batch_dims = rotmat.shape[:-2]

    # transformation matrix
    I = torch.eye(4, dtype=torch.float32, device=rotmat.device)
    I = I.repeat(batch_dims + (1, 1))

    # fill rotation matrix
    I[..., :3, :3] = rotmat

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

def from_ortho6d(r):
    return ortho6d.to_rotmat(r)

def from_xform(x):
    return xform.to_rotmat(x)