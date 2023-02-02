import torch
import torch.nn.functional as F
import numpy as np

from pymovis.ops import mathops

"""
Rotation conversions and operations

Glossary:
- A: Axis angle
- E: Euler angles
- R: Rotation matrix
- R6: 6D rotation vector [Zhou et al. 2018]
- Q: Quaternion (order in (w, x, y, z), where w is real value)
- v: Vector
- p: Position
"""

""" Operations with R """
def R_to_R6_torch(R):
    return R[..., :2, :].clone().reshape(R.shape[:-2] + (6,))

def R_to_R6_numpy(R):
    return R[..., :2, :].copy().reshape(R.shape[:-2] + (6,))

def R_to_R6(R):
    if isinstance(R, torch.Tensor):
        return R_to_R6_torch(R)
    elif isinstance(R, np.ndarray):
        return R_to_R6_numpy(R)
    else:
        raise TypeError(f"Type must be torch.Tensor or numpy.ndarray, but got {type(R)}")

####################################################################################

def R_inv_torch(R):
    return R.transpose(-1, -2)

def R_inv_numpy(R):
    return np.transpose(R, axes=[*range(len(R.shape) - 2), -1, -2])

def R_inv(R):
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R.shape must be (..., 3, 3), but got {R.shape}")

    if isinstance(R, torch.Tensor):
        return R_inv_torch(R)
    elif isinstance(R, np.ndarray):
        return R_inv_numpy(R)
    else:
        raise TypeError(f"Type must be torch.Tensor or numpy.ndarray, but got {type(R)}")

""" Operations with E """
def E_to_R_torch(E, order, radians=True):
    if not radians:
        E = torch.deg2rad(E)

    def _euler_axis_to_R(angle, axis):
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)
        cos = torch.cos(angle)
        sin = torch.sin(angle)

        if axis == "x":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        else:
            raise ValueError(f"Invalid axis: {axis}")

        return torch.stack(R_flat, dim=-1).reshape(angle.shape + (3, 3))
    
    Rs = [_euler_axis_to_R(E[..., i], order[i]) for i in range(3)]
    return torch.matmul(torch.matmul(Rs[0], Rs[1]), Rs[2])

def E_to_R_numpy(E, order, radians=True):
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

def E_to_R(E, order, radians=True):
    """
    Args:
        E: (..., 3)
    """
    if E.shape[-1] != 3:
        raise ValueError(f"E.shape must be (..., 3), but got {E.shape}")
    if len(order) != 3:
        raise ValueError(f"Order must have 3 characters, but got {order}")

    if isinstance(E, torch.Tensor):
        return E_to_R_torch(E, order, radians)
    elif isinstance(E, np.ndarray):
        return E_to_R_numpy(E, order, radians)
    else:
        raise TypeError(f"Type must be torch.Tensor or numpy.ndarray, but got {type(E)}")

####################################################################################

def E_to_Q_torch(E, order, radians=True):
    if not radians:
        E = torch.deg2rad(E)
    
    def _euler_axis_to_Q(angle, axis):
        zero = torch.zeros_like(angle)
        cos = torch.cos(angle / 2)
        sin = torch.sin(angle / 2)

        if axis == "x":
            Q_flat = (cos, sin, zero, zero)
        elif axis == "y":
            Q_flat = (cos, zero, sin, zero)
        elif axis == "z":
            Q_flat = (cos, zero, zero, sin)
        else:
            raise ValueError(f"Invalid axis: {axis}")
        return torch.stack(Q_flat, dim=-1).reshape(angle.shape + (4,))
        
    Qs = [_euler_axis_to_Q(E[..., i], order[i]) for i in range(3)]
    return Q_mul_torch(Q_mul_torch(Qs[0], Qs[1]), Qs[2])

def E_to_Q_numpy(E, order, radians=True):
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
    return Q_mul_numpy(Q_mul_numpy(Q[0], Q[1]), Q[2])

def E_to_Q(E, order, radians=True):
    """
    Args:
        E: (..., 3)
    """
    if E.shape[-1] != 3:
        raise ValueError(f"E.shape must be (..., 3), but got {E.shape}")
    if len(order) != 3:
        raise ValueError(f"Order must have 3 characters, but got {order}")

    if isinstance(E, torch.Tensor):
        return E_to_Q_torch(E, order, radians)
    elif isinstance(E, np.ndarray):
        return E_to_Q_numpy(E, order, radians)
    else:
        raise TypeError(f"Type must be torch.Tensor or numpy.ndarray, but got {type(E)}")

""" Operations with A """
def A_to_R_torch(angle, axis):
    a0, a1, a2 = axis[..., 0], axis[..., 1], axis[..., 2]
    zero       = torch.zeros_like(a0)

    # skew symmetric matrix
    S   = torch.stack([zero, -a2, a1, a2, zero, -a0, -a1, a0, zero], dim=-1)
    S   = S.reshape(angle.shape + (3, 3)) # (..., 3, 3)

    # rotation matrix
    I   = torch.eye(3, dtype=torch.float32, device=angle.device) # (3, 3)
    I   = I.reshape(1, 3, 3).expand(angle.shape + (3, 3))        # (..., 3, 3)
    sin = torch.sin(angle)                                       # (...,)
    cos = torch.cos(angle)                                       # (...,)
    
    return I + S * sin[..., None, None] + torch.matmul(S, S) * (1 - cos[..., None, None]) # (..., 3, 3)

def A_to_R_numpy(angle, axis):
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

def A_to_R(angle, axis):
    """
    Args:
        angle: (...)
        axis:  (..., 3)
    Returns:
        Rotation matrix (..., 3, 3)
    """
    if angle.shape != axis.shape[:-1]:
        raise ValueError(f"angle.shape must be axis.shape[:-1], but got {angle.shape} and {axis.shape}")

    if isinstance(angle, torch.Tensor):
        return A_to_R_torch(angle, axis)
    elif isinstance(angle, np.ndarray):
        return A_to_R_numpy(angle, axis)
    else:
        raise TypeError(f"Type must be torch.Tensor or numpy.ndarray, but got {type(angle)}")
####################################################################################

def A_to_Q_torch(angle, axis):
    cos = torch.cos(angle / 2)
    sin = torch.sin(angle / 2)
    axis_sin = axis * sin[..., None]

    return torch.cat([cos[..., None], axis_sin], dim=-1) # (..., 4)

def A_to_Q_numpy(angle, axis):
    cos = np.cos(angle / 2)
    sin = np.sin(angle / 2)
    axis_sin = axis * sin[..., None]

    return np.concatenate([cos[..., None], axis_sin], axis=-1) # (..., 4)

def A_to_Q(angle, axis):
    """
    Args:
        angle: (...)
        axis:  (..., 3)
    Returns:
        Quaternion (..., 4)
    """
    if axis.shape[-1] != 3:
        raise ValueError(f"axis.shape must be (..., 3), but got {axis.shape}")
    if angle.shape != axis.shape[:-1]:
        raise ValueError(f"angle.shape must be axis.shape[:-1], but got {angle.shape} and {axis.shape}")

    if isinstance(angle, torch.Tensor):
        return A_to_Q_torch(angle, axis)
    elif isinstance(angle, np.ndarray):
        return A_to_Q_numpy(angle, axis)
    else:
        raise TypeError(f"Type must be torch.Tensor or numpy.ndarray, but got {type(angle)}")

""" Operations with Q """
def Q_to_A_torch(Q, eps=1e-8):
    axis, angle = torch.empty_like(Q[..., 1:]), torch.empty_like(Q[..., 0])

    length = torch.sqrt(torch.sum(Q[..., 1:] * Q[..., 1:], dim=-1)) # (...,)
    small_angles = length < eps

    angle[small_angles] = 0.0
    axis[small_angles]  = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=Q.device) # (..., 3)

    angle[~small_angles] = 2.0 * torch.atan2(length[~small_angles], Q[..., 0][~small_angles]) # (...,)
    axis[~small_angles]  = Q[..., 1:][~small_angles] / length[~small_angles][..., None] # (..., 3)
    return angle, axis

def Q_to_A_numpy(Q, eps=1e-8):
    axis, angle = np.empty_like(Q[..., 1:]), np.empty_like(Q[..., 0])

    length = np.sqrt(np.sum(Q[..., 1:] * Q[..., 1:], axis=-1)) # (...,)
    small_angles = length < eps

    angle[small_angles] = 0.0
    axis[small_angles]  = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    angle[~small_angles] = 2.0 * np.arctan2(length[~small_angles], Q[..., 0][~small_angles]) # (...,)
    axis[~small_angles]  = Q[..., 1:][~small_angles] / length[~small_angles][..., None] # (..., 3)
    return angle, axis

def Q_to_A(Q, eps=1e-8):
    """
    Args:
        Q: (..., 4)
    Returns:
        angle: (...)
        axis:  (..., 3)
    """
    if Q.shape[-1] != 4:
        raise ValueError(f"Q.shape must be (..., 4), but got {Q.shape}")

    if isinstance(Q, torch.Tensor):
        return Q_to_A_torch(Q, eps=eps)
    elif isinstance(Q, np.ndarray):
        return Q_to_A_numpy(Q, eps=eps)
    else:
        raise TypeError(f"Type must be torch.Tensor or numpy.ndarray, but got {type(Q)}")

####################################################################################

def Q_to_R_torch(Q):
    two_s = 2.0 / torch.sum(Q * Q, dim=-1) # (...,)
    r, i, j, k = Q[..., 0], Q[..., 1], Q[..., 2], Q[..., 3]

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
    return R.reshape(Q.shape[:-1] + (3, 3)) # (..., 3, 3)

def Q_to_R_numpy(Q):
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

def Q_to_R(Q):
    """
    Args:
        Q: (..., 4)
    Returns:
        R: (..., 3, 3)
    """
    if Q.shape[-1] != 4:
        raise ValueError(f"Q.shape must be (..., 4), but got {Q.shape}")

    if isinstance(Q, torch.Tensor):
        return Q_to_R_torch(Q)
    elif isinstance(Q, np.ndarray):
        return Q_to_R_numpy(Q)
    else:
        raise TypeError(f"Type must be torch.Tensor or numpy.ndarray, but got {type(Q)}")
    
####################################################################################

def Q_to_R6_torch(Q):
    R = Q_to_R_torch(Q)
    return torch.cat([R[..., 0, :], R[..., 1, :]], dim=-1)

def Q_to_R6_numpy(Q):
    R = Q_to_R_numpy(Q)
    return np.concatenate([R[..., 0, :], R[..., 1, :]], axis=-1)

def Q_to_R6(Q):
    """
    Args:
        Q: (..., 4)
    Returns:
        R6: (..., 6)
    """
    if Q.shape[-1] != 4:
        raise ValueError(f"Q.shape must be (..., 4), but got {Q.shape}")

    if isinstance(Q, torch.Tensor):
        return Q_to_R6_torch(Q)
    elif isinstance(Q, np.ndarray):
        return Q_to_R6_numpy(Q)
    else:
        raise TypeError(f"Type must be torch.Tensor or numpy.ndarray, but got {type(Q)}")

####################################################################################

def Q_mul_torch(Q0, Q1):
    r0, i0, j0, k0 = torch.split(Q0, 1, dim=-1)
    r1, i1, j1, k1 = torch.split(Q1, 1, dim=-1)

    res = torch.cat([
        r0*r1 - i0*i1 - j0*j1 - k0*k1,
        r0*i1 + i0*r1 + j0*k1 - k0*j1,
        r0*j1 - i0*k1 + j0*r1 + k0*i1,
        r0*k1 + i0*j1 - j0*i1 + k0*r1
    ], dim=-1)

    return res

def Q_mul_numpy(Q0, Q1):
    r0, i0, j0, k0 = np.split(Q0, 4, axis=-1)
    r1, i1, j1, k1 = np.split(Q1, 4, axis=-1)

    res = np.concatenate([
        r0*r1 - i0*i1 - j0*j1 - k0*k1,
        r0*i1 + i0*r1 + j0*k1 - k0*j1,
        r0*j1 - i0*k1 + j0*r1 + k0*i1,
        r0*k1 + i0*j1 - j0*i1 + k0*r1
    ], axis=-1)

    return res

def Q_mul(Q0, Q1):
    """
    Args:
        Q0: left-hand quaternion (..., 4)
        Q1: right-hand quaternion (..., 4)
    Returns:
        Q: quaternion product Q0 * Q1 (..., 4)
    """
    if Q0.shape[-1] != 4 or Q1.shape[-1] != 4:
        raise ValueError(f"Q0.shape and Q1.shape must be (..., 4), but got {Q0.shape} and {Q1.shape}")

    if isinstance(Q0, torch.Tensor):
        return Q_mul_torch(Q0, Q1)
    elif isinstance(Q0, np.ndarray):
        return Q_mul_numpy(Q0, Q1)
    else:
        raise TypeError(f"Type must be torch.Tensor or numpy.ndarray, but got {type(Q0)}")

####################################################################################

def Q_inv_torch(Q):
    res = torch.tensor([1, -1, -1, -1], dtype=torch.float32, device=Q.device) * Q
    return res

def Q_inv_numpy(Q):
    res = np.array([1, -1, -1, -1], dtype=np.float32) * Q
    return res

def Q_inv(Q):
    """
    Args:
        Q: (..., 4)
    Returns:
        Q_inv: (..., 4)
    """
    if Q.shape[-1] != 4:
        raise ValueError(f"Q.shape must be (..., 4), but got {Q.shape}")

    if isinstance(Q, torch.Tensor):
        return Q_inv_torch(Q)
    elif isinstance(Q, np.ndarray):
        return Q_inv_numpy(Q)
    else:
        raise TypeError(f"Type must be torch.Tensor or numpy.ndarray, but got {type(Q)}")

""" Operations with R6 """
def R6_to_R_torch(R6):
    x_, y_ = R6[..., :3], R6[..., 3:]
    x = F.normalize(x_, dim=-1)
    y = F.normalize(y_ - (x * y_).sum(dim=-1, keepdim=True) * x, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], dim=-2) # (..., 3, 3)

def R6_to_R_numpy(R6):
    x_, y_ = R6[..., :3], R6[..., 3:]
    x = mathops.normalize_numpy(x_, axis=-1)
    y = mathops.normalize_numpy(y_ - np.sum(x * y_, axis=-1, keepdims=True) * x, axis=-1)
    z = np.cross(x, y, axis=-1)
    return np.stack([x, y, z], axis=-2) # (..., 3, 3)

def R6_to_R(R6):
    """
    Args:
        R6: (..., 6)
    Returns:
        R: (..., 3, 3)
    """
    if R6.shape[-1] != 6:
        raise ValueError(f"R6.shape must be (..., 6), but got {R6.shape}")

    if isinstance(R6, torch.Tensor):
        return R6_to_R_torch(R6)
    elif isinstance(R6, np.ndarray):
        return R6_to_R_numpy(R6)
    else:
        raise TypeError(f"Type must be torch.Tensor or numpy.ndarray, but got {type(R6)}")