import torch
import torch.nn.functional as F
from . import rotmat, aaxis, euler, ortho6d, xform

"""
Quaternion operations
"""
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
    t = 2.0 * torch.cross(q[..., 1:], v, dim=-1)
    res = v + q[..., 0:1] * t + torch.cross(q[..., 1:], t, dim=-1)
    return res

def inv(q):
    return torch.cat([q[..., 0:1], -q[..., 1:]], dim=-1)

def identity(device="cpu"):
    return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

def interpolate(q_from, q_to, t):
    """
    Args:
        q_from: (..., 4)
        q_to: (..., 4)
        t: (..., t) or (t,), or just a float
    Returns:
        interpolated quaternion (..., 4, t)
    """
    device = q_from.device
    
    # ensure t is a torch tensor
    if isinstance(t, float):
        t = torch.tensor([t], dtype=torch.float32, device=device)
    t = torch.zeros_like(q_from[..., 0:1]) + t # (..., t)
    
    # ensure unit quaternions
    q_from_ = F.normalize(q_from, dim=-1, eps=1e-8) # (..., 4)
    q_to_   = F.normalize(q_to,   dim=-1, eps=1e-8) # (..., 4)

    # ensure positive dot product
    dot = torch.sum(q_from_ * q_to_, dim=-1) # (...,)
    neg = dot < 0.0
    dot[neg] = -dot[neg]
    q_to_[neg] = -q_to_[neg]

    # omega = arccos(dot)
    linear = dot > 0.9999
    omegas = torch.acos(dot[~linear]) # (...,)
    omegas = omegas[..., None] # (..., 1)
    sin_omegas = torch.sin(omegas) # (..., 1)

    # interpolation amounts
    t0 = torch.empty_like(t)
    t0[linear] = 1.0 - t[linear]
    t0[~linear] = torch.sin((1.0 - t[~linear]) * omegas) / sin_omegas # (..., t)

    t1 = torch.empty_like(t)
    t1[linear] = t[linear]
    t1[~linear] = torch.sin(t[~linear] * omegas) / sin_omegas # (..., t)
    
    # interpolate
    q_interp = t0[..., None, :] * q_from_[..., :, None] + t1[..., None, :] * q_to_[..., :, None] # (..., 4, t)
    q_interp = F.normalize(q_interp, dim=-2, eps=1e-8) # (..., 4, t)
    
    return q_interp

def between_vecs(v_from, v_to):
    v_from_ = F.normalize(v_from, dim=-1, eps=1e-8) # (..., 3)
    v_to_   = F.normalize(v_to,   dim=-1, eps=1e-8) # (..., 3)

    dot = torch.sum(v_from_ * v_to_, dim=-1) # (...,)
    cross = torch.cross(v_from_, v_to_)
    cross = F.normalize(cross, dim=-1, eps=1e-8) # (..., 3)
    
    real = torch.sqrt(0.5 * (1.0 + dot))
    imag = torch.sqrt(0.5 * (1.0 - dot))[..., None] * cross

    return torch.cat([real[..., None], imag], dim=-1)

def fk(local_quats, root_pos, skeleton):
    """
    Attributes:
        local_quats: (..., J, 4)
        root_pos: (..., 3), global root position
        skeleton: aPyOpenGL.agl.Skeleton
    """
    pre_xforms = torch.from_numpy(skeleton.pre_xforms).to(local_quats.device)
    pre_xforms = torch.tile(pre_xforms, local_quats.shape[:-2] + (1, 1, 1)) # (..., J, 4, 4)
    pre_quats  = xform.to_quat(pre_xforms)
    pre_pos    = xform.to_translation(pre_xforms)
    pre_pos[..., 0, :] = root_pos

    global_quats = [mul(pre_quats[..., 0, :], local_quats[..., 0, :])]
    global_pos = [pre_pos[..., 0, :]]

    for i in range(1, skeleton.num_joints):
        parent_idx = skeleton.parent_idx[i]
        global_quats.append(mul(mul(global_quats[parent_idx], pre_quats[..., i, :]), local_quats[..., i, :]))
        global_pos.append(mul_vec(global_quats[parent_idx], pre_pos[..., i, :]) + global_pos[parent_idx])
    
    global_quats = torch.stack(global_quats, dim=-2) # (..., J, 4)
    global_pos = torch.stack(global_pos, dim=-2) # (..., J, 3)

    return global_quats, global_pos

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

    rotmat = torch.stack([
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
    return rotmat.reshape(quat.shape[:-1] + (3, 3)) # (..., 3, 3)

def to_ortho6d(quat):
    return rotmat.to_ortho6d(to_rotmat(quat))

def to_xform(quat):
    return rotmat.to_xform(to_rotmat(quat))

"""
Other representations to quaternion
"""
def from_aaxis(a):
    return aaxis.to_quat(a)

def from_euler(angles, order, radians=True):
    return euler.to_quat(angles, order, radians=radians)

def from_rotmat(r):
    return rotmat.to_quat(r)

def from_ortho6d(r6d):
    return ortho6d.to_quat(r6d)