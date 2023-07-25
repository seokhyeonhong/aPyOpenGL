import numpy as np

from . import rotmat, aaxis, euler, ortho6d, xform

"""
Quaternion operations
"""
def mul(q0, q1):
    r0, i0, j0, k0 = np.split(q0, 4, axis=-1)
    r1, i1, j1, k1 = np.split(q1, 4, axis=-1)

    res = np.concatenate([
        r0*r1 - i0*i1 - j0*j1 - k0*k1,
        r0*i1 + i0*r1 + j0*k1 - k0*j1,
        r0*j1 - i0*k1 + j0*r1 + k0*i1,
        r0*k1 + i0*j1 - j0*i1 + k0*r1
    ], axis=-1)

    return res

def mul_vec(q, v):
    t = 2.0 * np.cross(q[..., 1:], v)
    res = v + q[..., 0:1] * t + np.cross(q[..., 1:], t)
    return res

def inv(q):
    return np.concatenate([q[..., 0:1], -q[..., 1:]], axis=-1)

def identity():
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

def interpolate(q_from, q_to, t):
    len = np.sum(q_from * q_to, axis=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    q_to[neg] = -q_to[neg]

    t = np.zeros_like(q_from[..., 0:1]) + t
    t0 = np.zeros(t.shape, dtype=np.float32)
    t1 = np.zeros(t.shape, dtype=np.float32)

    linear = (1.0 - t) < 0.01
    omegas = np.arccos(len[~linear])
    sin_omegas = np.sin(omegas)

    t0[linear] = 1.0 - t[linear]
    t0[~linear] = np.sin((1.0 - t[~linear]) * omegas) / sin_omegas

    t1[linear] = t[linear]
    t1[~linear] = np.sin(t[~linear] * omegas) / sin_omegas
    res = t0 * q_from + t1 * q_to
    
    return res

def between_vecs(v_from, v_to):
    v_from_ = v_from / (np.linalg.norm(v_from, axis=-1, keepdims=True) + 1e-8) # (..., 3)
    v_to_   = v_to / (np.linalg.norm(v_to,   axis=-1, keepdims=True) + 1e-8)   # (..., 3)

    dot = np.sum(v_from_ * v_to_, axis=-1) # (...,)
    cross = np.cross(v_from_, v_to_)
    cross = cross / (np.linalg.norm(cross, axis=-1, keepdims=True) + 1e-8) # (..., 3)
    
    real = np.sqrt((1.0 + dot) * 0.5) # (...,)
    imag = np.sqrt((1.0 - dot) * 0.5)[..., None] * cross
    
    return np.concatenate([real[..., None], imag], axis=-1)

def fk(local_quats, root_pos, skeleton):
    """
    Attributes:
        local_quats: (..., J, 4)
        root_pos: (..., 3), global root position
        skeleton: aPyOpenGL.agl.Skeleton
    """
    pre_xforms = np.tile(skeleton.pre_xforms, local_quats.shape[:-2] + (1, 1, 1)) # (..., J, 4, 4)
    pre_quats  = xform.to_quat(pre_xforms) # (..., J, 4)
    pre_pos    = xform.to_translation(pre_xforms) # (..., J, 3)
    pre_pos[..., 0, :] = root_pos

    global_quats = [mul(pre_quats[..., 0, :], local_quats[..., 0, :])]
    global_pos = [pre_pos[..., 0, :]]

    for i in range(1, skeleton.num_joints):
        parent_idx = skeleton.parent_idx[i]
        global_quats.append(mul(mul(global_quats[parent_idx], pre_quats[..., i, :]), local_quats[..., i, :]))
        global_pos.append(mul_vec(global_quats[parent_idx], pre_pos[..., i, :]) + global_pos[parent_idx])
    
    global_quats = np.stack(global_quats, axis=-2) # (..., J, 4)
    global_pos = np.stack(global_pos, axis=-2) # (..., J, 3)

    return global_quats, global_pos

"""
Quaternion to other representations
"""
def to_aaxis(quat):
    axis, angle = np.empty_like(quat[..., 1:]), np.empty_like(quat[..., 0])

    # small angles
    length = np.sqrt(np.sum(quat[..., 1:] * quat[..., 1:], axis=-1)) # (...,)
    small_angles = length < 1e-8

    # avoid division by zero
    angle[small_angles] = 0.0
    axis[small_angles]  = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # normal case
    angle[~small_angles] = 2.0 * np.arctan2(length[~small_angles], quat[..., 0][~small_angles]) # (...,)
    axis[~small_angles]  = quat[..., 1:][~small_angles] / length[~small_angles][..., None] # (..., 3)

    # make sure angle is in [-pi, pi)
    large_angles = angle >= np.pi
    angle[large_angles] = angle[large_angles] - 2 * np.pi

    return axis * angle[..., None] # (..., 3)

def to_rotmat(quat):
    two_s = 2.0 / np.sum(quat * quat, axis=-1) # (...,)
    r, i, j, k = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    rotmat = np.stack([
        1.0 - two_s * (j*j + k*k),
        two_s * (i*j - k*r),
        two_s * (i*k + j*r),
        two_s * (i*j + k*r),
        1.0 - two_s * (i*i + k*k),
        two_s * (j*k - i*r),
        two_s * (i*k - j*r),
        two_s * (j*k + i*r),
        1.0 - two_s * (i*i + j*j)
    ], axis=-1)
    return rotmat.reshape(quat.shape[:-1] + (3, 3)) # (..., 3, 3)

def to_ortho6d(quat):
    return rotmat.to_ortho6d(to_rotmat(quat))

def to_xform(quat, translation=None):
    return rotmat.to_xform(to_rotmat(quat), translation=translation)

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

def from_xform(x):
    return xform.to_quat(x)