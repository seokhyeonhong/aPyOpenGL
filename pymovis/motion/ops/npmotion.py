import numpy as np

"""
Functions that convert between different rotation representations.

Glossary:
- A: Axis angle
- E: Euler angles
- R: Rotation matrix
- R6: 6D rotation vector [Zhou et al. 2018]
- Q: Quaternion (order in (w, x, y, z), where w is real value)
- v: Vector
- p: Position

TODO: Refactor code & Synchronize with torchmotion.py
"""

def normalize_vector(x, axis=-1, eps=1e-8):
    res = x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)
    return res

""" FK """
def R_fk(local_R, root_p, skeleton):
    """
    Args:
        local_R: (..., N, 3, 3)
        root_p: (..., 3)
        bone_offset: (N, 3)
        parents: (N,)
    """
    bone_offsets, parents = skeleton.get_bone_offsets(), skeleton.parent_idx

    global_R, global_p = [local_R[..., 0, :, :]], [root_p]
    for i in range(1, len(parents)):
        global_R.append(np.matmul(global_R[parents[i]], local_R[..., i, :, :]))
        global_p.append(np.matmul(global_R[parents[i]], bone_offsets[i]) + global_p[parents[i]])
    
    global_R = np.stack(global_R, axis=-3) # (..., N, 3, 3)
    global_p = np.stack(global_p, axis=-2) # (..., N, 3)
    return global_R, global_p

""" Operations with R """
def R_to_R6(R):
    """
    Args:
        R: (..., 3, 3)
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Invalid rotation matrix shape {R.shape}")
    return R[..., :2, :].copy().reshape(R.shape[:-2] + (6,))

def R_inv(R):
    """
    Args:
        R: (..., N, 3, 3)
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Invalid rotation matrix shape {R.shape}")

    return np.transpose(R, axes=[*range(len(R.shape) - 2), -1, -2])

""" Operations with E """
def E_to_R(E, order, radians=True):
    """
    Args:
        E: (..., 3)
    """
    if E.shape[-1] != 3:
        raise ValueError(f"Invalid Euler angles shape {E.shape}")
    if len(order) != 3:
        raise ValueError(f"Order must have 3 characters, but got {order}")

    if not radians:
        E = np.deg2rad(E)

    def _euler_axis_to_R(angle, axis):
        one  = np.ones_like(angle, dtype=np.float32)
        zero = np.zeros_like(angle, dtype=np.float32)
        cos  = np.cos(angle, dtype=np.float32)
        sin  = np.sin(angle, dtype=np.float32)

        if axis == "x":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        else:
            raise ValueError(f"Invalid axis: {axis}")
        return np.stack(R_flat, axis=-1).reshape(angle.shape + (3, 3))

    R = [_euler_axis_to_R(E[..., i], order[i]) for i in range(3)]
    return np.matmul(np.matmul(R[0], R[1]), R[2])

def E_to_Q(E, order, radians=True):
    """
    Args:
        E: (..., 3)
    """
    if E.shape[-1] != 3:
        raise ValueError(f"Invalid Euler angles shape {E.shape}")
    if len(order) != 3:
        raise ValueError(f"Order must have 3 characters, but got {order}")

    if not radians:
        E = np.deg2rad(E)
    
    def _euler_axis_to_Q(angle, axis):
        zero = np.zeros_like(angle, dtype=np.float32)
        cos  = np.cos(angle / 2, dtype=np.float32)
        sin  = np.sin(angle / 2, dtype=np.float32)

        if axis == "x":
            Q_flat = (cos, sin, zero, zero)
        elif axis == "y":
            Q_flat = (cos, zero, sin, zero)
        elif axis == "z":
            Q_flat = (cos, zero, zero, sin)
        else:
            raise ValueError(f"Invalid axis: {axis}")
        return np.stack(Q_flat, axis=-1).reshape(angle.shape + (4,))

    Q = [_euler_axis_to_Q(E[..., i], order[i]) for i in range(3)]
    return Q_mul(Q_mul(Q[0], Q[1]), Q[2])

""" Operations with A """
def A_to_R(angle, axis):
    """
    Args:
        angle: (...)
        axis:  (..., 3)
    Returns:
        Rotation matrix (..., 3, 3)
    """
    if axis.shape[-1] != 3:
        raise ValueError(f"Invalid axis shape {axis.shape}")
    if angle.shape != axis.shape[:-1]:
        raise ValueError(f"Incompatible angle shape {angle.shape} and and axis shape {axis.shape}")

    a0, a1, a2 = axis[..., 0], axis[..., 1], axis[..., 2]
    zero       = np.zeros_like(a0)

    # skew symmetric matrix
    S   = np.stack([zero, -a2, a1, a2, zero, -a0, -a1, a0, zero], axis=-1)
    S   = S.reshape(angle.shape + (3, 3))             # (..., 3, 3)

    # rotation matrix
    I   = np.eye(3, dtype=np.float32)                 # (3, 3)
    I   = np.tile(I, reps=(angle.shape + (1, 1)))     # (..., 3, 3)
    sin = np.sin(angle)                               # (...,)
    cos = np.cos(angle)                               # (...,)

    return I + S * sin + np.matmul(S, S) * (1 - cos)  # (..., 3, 3)

def A_to_Q(angle, axis):
    """
    Args:
        angle: angles tensor (...)
        axis: axis tensor (..., 3)
    Returns:
        Quaternion tensor (..., 4)
    """
    if axis.shape[-1] != 3:
        raise ValueError(f"Invalid axis shape {axis.shape}")
    if angle.shape != axis.shape[:-1]:
        raise ValueError(f"Incompatible angle shape {angle.shape} and and axis shape {axis.shape}")

    cos = np.cos(angle / 2)
    sin = np.sin(angle / 2)
    axis_sin = axis * sin[..., None]

    return np.concatenate([cos[..., None], axis_sin], axis=-1) # (..., 4)

""" Operations with Q """
def Q_to_A(Q, eps=1e-8):
    """
    Args:
        Q: quaternion tensor (..., 4)
    Returns:
        angle: angle tensor (...)
        axis:  axis tensor (..., 3)
    """
    if Q.shape[-1] != 4:
        raise ValueError(f"Invalid quaternion shape {Q.shape}")

    axis, angle = np.empty_like(Q[..., 1:]), np.empty_like(Q[..., 0])

    length = np.sqrt(np.sum(Q[..., 1:] * Q[..., 1:], axis=-1)) # (...,)
    small_angles = length < eps

    angle[small_angles] = 0.0
    axis[small_angles]  = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    angle[~small_angles] = 2.0 * np.arctan2(length[~small_angles], Q[..., 0][~small_angles]) # (...,)
    axis[~small_angles]  = Q[..., 1:][~small_angles] / length[~small_angles][..., None] # (..., 3)
    return angle, axis

def Q_to_R(Q):
    """
    Args:
        Q: (..., 4)
    Returns:
        R: (..., 3, 3)
    """
    if Q.shape[-1] != 4:
        raise ValueError(f"Invalid quaternion shape {Q.shape}")
    
    two_s = 2.0 / np.sum(Q * Q, axis=-1) # (...,)
    r, i, j, k = Q[..., 0], Q[..., 1], Q[..., 2], Q[..., 3]

    R = np.stack([
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
    return R.reshape(Q.shape[:-1] + (3, 3)) # (..., 3, 3)

def Q_to_R6(Q):
    """
    Args:
        Q: (..., 4)
    """
    if Q.shape[-1] != 4:
        raise ValueError(f"Invalid quaternion shape {Q.shape}")
    
    R = Q_to_R(Q)
    return np.concatenate([R[..., 0, :], R[..., 1, :]], axis=-1)

def Q_mul(Q0, Q1):
    """
    Args:
        Q0: left-hand quaternion (..., 4)
        Q1: right-hand quaternion (..., 4)
    Returns:
        Q: quaternion product Q0 * Q1 (..., 4)
    """
    if Q0.shape[-1] != 4 or Q1.shape[-1] != 4:
        raise ValueError(f"Incompatible shapes {Q0.shape} and {Q1.shape}")
    
    r0, i0, j0, k0 = np.split(Q0, 4, axis=-1)
    r1, i1, j1, k1 = np.split(Q1, 4, axis=-1)

    res = np.concatenate([
        r0*r1 - i0*i1 - j0*j1 - k0*k1,
        r0*i1 + i0*r1 + j0*k1 - k0*j1,
        r0*j1 - i0*k1 + j0*r1 + k0*i1,
        r0*k1 + i0*j1 - j0*i1 + k0*r1
    ], axis=-1)

    return res

def Q_inv(Q):
    """
    Args:
        Q: (..., 4)
    """
    if Q.shape[-1] != 4:
        raise ValueError(f"Invalid quaternion shape {Q.shape}")

    res = np.array([1, -1, -1, -1], dtype=np.float32) * Q
    return res

""" Operations with R6 """
def R6_to_R(R6):
    """
    Args:
        R6: (..., 6)
    """
    if R6.shape[-1] != 6:
        raise ValueError(f"Invalid R6 shape {R6.shape}")
    
    x = normalize_vector(R6[..., 0:3], axis=-1)
    y = normalize_vector(R6[..., 3:6] - np.sum(x * R6[..., 3:6], axis=-1, keepdims=True) * x, axis=-1)
    z = np.cross(x, y, axis=-1)
    return np.stack([x, y, z], axis=-2) # (..., 3, 3)