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
def R_fk(R, root_p, skeleton):
    """
    :param R: (..., N, 3, 3)
    :param root_p: (..., 3)
    :param bone_offset: (N, 3)
    :param parents: (N,)
    """
    bone_offsets, parents = skeleton.get_bone_offsets(), skeleton.parent_idx

    global_R, global_p = [R[..., 0, :, :]], [root_p]
    for i in range(1, len(parents)):
        global_R.append(np.matmul(global_R[parents[i]], R[..., i, :, :]))
        global_p.append(np.matmul(global_R[parents[i]], bone_offsets[i]) + global_p[parents[i]])
    
    global_R = np.stack(global_R, axis=-3) # (..., N, 3, 3)
    global_p = np.stack(global_p, axis=-2) # (..., N, 3)
    return global_R, global_p

""" Conversion from R """
def R_to_R6(R):
    """
    Parameters
        R: (..., 3, 3)
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R.shape[-2:] = {R.shape[-2:]} != (3, 3)")
    r0 = R[..., 0, :]
    r1 = R[..., 1, :]
    return np.concatenate([r0, r1], axis=-1) # (..., 6)

""" Conversion from E """
def E_to_R(E, order, radians=True):
    """
    Parameters
        E: (..., 3)
    """
    if not radians:
        E = np.deg2rad(E)

    R_map = {
        "x": lambda x: np.stack([np.ones_like(x), np.zeros_like(x), np.zeros_like(x),
                                np.zeros_like(x), np.cos(x), -np.sin(x),
                                np.zeros_like(x), np.sin(x), np.cos(x)], axis=-1).reshape(*x.shape, 3, 3),
        "y": lambda x: np.stack([np.cos(x), np.zeros_like(x), np.sin(x),
                                np.zeros_like(x), np.ones_like(x), np.zeros_like(x),
                                -np.sin(x), np.zeros_like(x), np.cos(x)], axis=-1).reshape(*x.shape, 3, 3),
        "z": lambda x: np.stack([np.cos(x), -np.sin(x), np.zeros_like(x),
                                np.sin(x), np.cos(x), np.zeros_like(x),
                                np.zeros_like(x), np.zeros_like(x), np.ones_like(x)], axis=-1).reshape(*x.shape, 3, 3),
    }

    if len(order) == 3:
        R0 = R_map[order[0]](E[..., 0])
        R1 = R_map[order[1]](E[..., 1])
        R2 = R_map[order[2]](E[..., 2])
        return R0 @ R1 @ R2
    elif len(order) == 1:
        return R_map[order](E)
    else:
        raise ValueError(f"Invalid order: {order}")

def E_to_Q(E, order):
    axis = {
        'x': np.asarray([1, 0, 0], dtype=np.float32),
        'y': np.asarray([0, 1, 0], dtype=np.float32),
        'z': np.asarray([0, 0, 1], dtype=np.float32)
    }

    q0 = A_to_Q(E[..., 0], axis[order[0]])
    q1 = A_to_Q(E[..., 1], axis[order[1]])
    q2 = A_to_Q(E[..., 2], axis[order[2]])

    return Q_mul(q0, Q_mul(q1, q2))

""" Conversion from A """
def A_to_R(angle, axis):
    """
    Parameters
        angle: (..., N)
        axis:  (..., 3)
    """
    if axis.shape[-1] != 3:
        raise ValueError(f"axis.shape[-1] = {axis.shape[-1]} != 3")
    
    if angle.shape == axis.shape[:-1]:
        angle = angle[..., np.newaxis]

    a0, a1, a2     = axis[..., 0], axis[..., 1], axis[..., 2]
    zero           = np.zeros_like(a0)
    skew_symmetric = np.stack([zero, -a2, a1,
                                a2, zero, -a0,
                                -a1, a0, zero], axis=-1).reshape(*angle.shape[:-1], 1, 3, 3) # (..., 1, 3, 3)

    I              = np.eye(3, dtype=np.float32)                                          # (3, 3)
    I              = np.tile(I, reps=[*angle.shape[:-1], 1, 1])[..., np.newaxis, :, :]    # (..., 1, 3, 3)
    sin            = np.sin(angle)[..., np.newaxis, np.newaxis]                           # (..., N, 1, 1)
    cos            = np.cos(angle)[..., np.newaxis, np.newaxis]                           # (..., N, 1, 1)

    return I + skew_symmetric * sin + np.matmul(skew_symmetric, skew_symmetric) * (1 - cos) # (..., N, 3, 3)

def A_to_Q(angle, axis):
    """
    Parameters
        angle: angles tensor (..., N)
        axis: axis tensor (..., 3)
    """
    if axis.shape[-1] != 3:
        raise ValueError(f"axis.shape[-1] = {axis.shape[-1]} != 3")

    axis = normalize_vector(axis, axis=-1)
    a0, a1, a2 = axis[..., 0], axis[..., 1], axis[..., 2]
    cos = np.cos(angle / 2)[..., np.newaxis]
    sin = np.sin(angle / 2)[..., np.newaxis]

    return np.concatenate([cos, a0 * sin, a1 * sin, a2 * sin], axis=-1) # (..., 4)

""" Conversion from Q """
def Q_to_R(q: np.ndarray) -> np.ndarray:
    """
    Parameters
        q: (..., 4)
    """
    if q.shape[-1] != 4:
        raise ValueError(f"q.shape[-1] = {q.shape[-1]} != 4")
    
    q = normalize_vector(q, axis=-1)
    w, x, y, z = np.split(q, 4, axis=-1)
    row0 = np.stack([2*(w*w + x*x) - 1, 2*(x*y - w*z), 2*(x*z + w*y)], axis=-1)
    row1 = np.stack([2*(w*z + x*y), 2*(w*w + y*y) - 1, 2*(y*z - w*x)], axis=-1)
    row2 = np.stack([2*(x*z - w*y), 2*(w*x + y*z), 2*(w*w + z*z) - 1], axis=-1)
    return np.stack([row0, row1, row2], axis=-2) # (..., 3, 3)

def Q_to_R6(Q):
    """
    Parameters
        Q: (..., 4)
    """
    if Q.shape[-1] != 4:
        raise ValueError(f"Q.shape[-1] = {Q.shape[-1]} != 4")
    
    Q = normalize_vector(Q, axis=-1)
    q0, q1, q2, q3 = Q[..., 0], Q[..., 1], Q[..., 2], Q[..., 3]

    r0 = np.stack([2*(q0*q0 + q1*q1) - 1, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)], axis=-1)
    r1 = np.stack([2*(q1*q2 + q0*q3), 2*(q0*q0 + q2*q2) - 1, 2*(q2*q3 - q0*q1)], axis=-1)
    return np.concatenate([r0, r1], axis=-1) # (..., 6)

""" Conversion from R6 """
def R6_to_R(R6: np.ndarray) -> np.ndarray:
    """
    Parameters
        R6: (..., 6)
    """
    if R6.shape[-1] != 6:
        raise ValueError(f"r6.shape[-1] = {R6.shape[-1]} != 6")
    
    x = normalize_vector(R6[..., 0:3], axis=-1)
    y = normalize_vector(R6[..., 3:6] - np.sum(x * R6[..., 3:6], axis=-1, keepdims=True) * x, axis=-1)
    z = np.cross(x, y, axis=-1)
    return np.stack([x, y, z], axis=-2) # (..., 3, 3)

""" Operations for R """
def R_inv(R):
    """
    Parameters
        R: (..., N, 3, 3)
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R.shape[-2:] = {R.shape[-2:]} != (3, 3)")
    return np.transpose(R, axes=[*range(len(R.shape) - 2), -1, -2])

""" Operations for Q """
def Q_mul(x, y):
    """
    Parameters
        x: tensor of quaternions of shape (..., Nb of joints, 4)
        y: tensor of quaternions of shape (..., Nb of joints, 4)
    """
    if x.shape[-1] != 4 or y.shape[-1] != 4:
        raise ValueError(f"x.shape[-1] = {x.shape[-1]} != 4 or y.shape[-1] = {y.shape[-1]} != 4")
        
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

    return res

def Q_inv(Q):
    """
    Parameters
        Q: (..., 4)
    """
    if Q.shape[-1] != 4:
        raise ValueError(f"Q.shape[-1] = {Q.shape[-1]} != 4")

    res = np.asarray([1, -1, -1, -1], dtype=np.float32) * Q
    return res