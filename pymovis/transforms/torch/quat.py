import torch
from . import rotmat

""" Quaternion operations """
def mul(q0, q1):
    r0, i0, j0, k0 = torch.split(q0, 1, dim=-1)
    r1, i1, j1, k1 = torch.split(q1, 1, dim=-1)
    
    res = torch.cat([
        r0*r1 - i0*i1 - j0*j1 - k0*k1,
        r0*i1 + i0*r1 + j0*k1 - k0*j1,
        r0*j1 - i0*k1 + j0*r1 + k0*i1,
        r0*k1 + i0*j1 - j0*i1 + k0*r1
    ], dim=-1)

    return res

def mul_vec(q, v):
    t = 2.0 * torch.cross(q[..., 1:], v)
    res = v + q[..., 0:1] * t + torch.cross(q[..., 1:], t)
    return res

def inv(q):
    return torch.cat([q[..., 0:1], -q[..., 1:]], dim=-1)

""" Quaternion to other representations """
def to_aaxis(quat):
    axis, angle = torch.empty_like(quat[..., 1:]), torch.empty_like(quat[..., 0])

    # small angles
    length = torch.sqrt(torch.sum(quat[..., 1:] * quat[..., 1:], dim=-1)) # (...,)
    small_angles = length < 1e-8

    # avoid division by zero
    angle[small_angles] = 0.0
    axis[small_angles]  = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=quat.device) # (..., 3)

    # normal case
    angle[~small_angles] = 2.0 * torch.atan2(length[~small_angles], quat[..., 0][~small_angles]) # (...,)
    axis[~small_angles]  = quat[..., 1:][~small_angles] / length[~small_angles][..., None] # (..., 3)

    # make sure angle is in [-pi, pi)
    large_angles = angle >= torch.pi
    angle[large_angles] = angle[large_angles] - 2 * torch.pi

    return axis * angle[..., None]

def to_rotmat(quat):
    two_s = 2.0 / torch.sum(quat * quat, dim=-1) # (...,)
    r, i, j, k = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    R = torch.stack([
        1.0 - two_s * (j*j + k*k),
        two_s * (i*j - k*r),
        two_s * (i*k + j*r),
        two_s * (i*j + k*r),
        1.0 - two_s * (i*i + k*k),
        two_s * (j*k - i*r),
        two_s * (i*k - j*r),
        two_s * (j*k + i*r),
        1.0 - two_s * (i*i + j*j)
    ], dim=-1)
    return R.reshape(quat.shape[:-1] + (3, 3)) # (..., 3, 3)

def to_rot6d(quat):
    return rotmat.to_rot6d(to_rotmat(quat))

def to_xform(quat):
    return rotmat.to_xform(to_rotmat(quat))
