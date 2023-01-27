import torch
import torch.nn.functional as F

from pymovis.motion.core import Skeleton

"""
Functions that convert between different rotation representations.

Glossary:
- A: Angle axis
- E: Euler angles
- R: Rotation matrix
- R6: 6D rotation vector
- Q: Quaternion (order in (w, x, y, z), where w is real value)
- v: Vector
- p: Position
"""

""" FK """
def R_fk(local_R, root_p, skeleton):
        """
        Args:
            local_R: (..., N, 3, 3)
            root_p: (..., 3)
            bone_offset: (N, 3)
            parents: (N,)
        Returns:
            Global rotation matrix and position of each joint.
        """
        bone_offsets = torch.from_numpy(skeleton.get_bone_offsets()).to(R.device)
        parents = skeleton.parent_idx

        global_R, global_p = [local_R[..., 0, :, :]], [root_p]
        for i in range(1, len(parents)):
            global_R.append(torch.matmul(global_R[parents[i]], local_R[..., i, :, :]))
            global_p.append(torch.matmul(global_R[parents[i]], bone_offsets[i]) + global_p[parents[i]])
        
        global_R = torch.stack(global_R, dim=-3) # (..., N, 3, 3)
        global_p = torch.stack(global_p, dim=-2) # (..., N, 3)
        return global_R, global_p


def R6_fk(local_R6, root_p, skeleton):
    """
    Args:
        local_R6: (..., N, 6)
        root_p: (..., 3)
        bone_offset: (N, 3)
        parents: (N,)
    """
    R, p = R_fk(R6_to_R(local_R6), root_p, skeleton)
    return R_to_R6(R), p

""" Operations with R """
def R_to_R6(R):
    """
    Parameters
        R: (..., 3, 3)
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Invalid rotation matrix shape {R.shape}")
    return R[..., :2, :].clone().reshape(R.shape[:-2] + (6,))

def R_inv(R):
        """
        Parameters
            R: (..., N, 3, 3)
        """
        if R.shape[-2:] != (3, 3):
            raise ValueError(f"Invalid rotation matrix shape {R.shape}")
        return R.transpose(-1, -2)

def R_mul(R0, R1):
    """
    Parameters
        R0: (..., N, 3, 3)
        R1: (..., N, 3, 3)
    """
    if R0.shape[-2:] != (3, 3) or R1.shape[-2:] != (3, 3):
        raise ValueError(f"Invalid rotation matrix shape {R0.shape} or {R1.shape}")
    return torch.matmul(R0, R1)

""" Conversion from E """
def E_to_R(E, order, radians=True):
    """
    Parameters
        E: (..., 3)
    """
    if E.shape[-1] != 3:
        raise ValueError(f"Invalid Euler angles shape {E.shape}")
    if len(order) != 3:
        raise ValueError(f"Order must have 3 characters, but got {order}")

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
    return Q_mul(Q_mul(Qs[0], Qs[1]), Qs[2])

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
    zero       = torch.zeros_like(a0)

    # skew symmetric matrix
    S   = torch.stack([zero, -a2, a1, a2, zero, -a0, -a1, a0, zero], dim=-1)
    S   = S.reshape(angle.shape + (3, 3))             # (..., 3, 3)

    # rotation matrix
    I   = torch.eye(3, dtype=torch.float32)                # (3, 3)
    I   = torch.tile(I, reps=(angle.shape + (1, 1)))     # (..., 3, 3)
    sin = torch.sin(angle)                               # (...,)
    cos = torch.cos(angle)                               # (...,)

    return I + S * sin + torch.matmul(S, S) * (1 - cos)  # (..., 3, 3)

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

    cos = torch.cos(angle / 2)
    sin = torch.sin(angle / 2)
    axis_sin = axis * sin[..., None]

    return torch.cat([cos[..., None], axis_sin], dim=-1) # (..., 4)

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

    axis, angle = torch.empty_like(Q[..., 1:]), torch.empty_like(Q[..., 0])

    length = torch.sqrt(torch.sum(Q[..., 1:] * Q[..., 1:], dim=-1)) # (...,)
    small_angles = length < eps

    angle[small_angles] = 0.0
    axis[small_angles]  = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=Q.device) # (..., 3)

    angle[~small_angles] = 2.0 * torch.atan2(length[~small_angles], Q[..., 0][~small_angles]) # (...,)
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

def Q_to_R6(Q):
    """
    Args:
        Q: (..., 4)
    """
    if Q.shape[-1] != 4:
        raise ValueError(f"Invalid quaternion shape {Q.shape}")
    
    R = Q_to_R(Q)
    return torch.cat([R[..., 0, :], R[..., 1, :]], dim=-1)

def Q_mul(Q0, Q1):
    """
    Args:
        Q0: left-hand quaternion (..., 4)
        Q1: right-hand quaternion (..., 4)
    Returns:
        Q: quaternion product Q0 * Q1 (..., 4)
    """
    if Q0.shape[-1] != 4 or Q1.shape[-1] != 4:
        raise ValueError(f"Invalid quaternion shape {Q0.shape} or {Q1.shape}")

    r0, i0, j0, k0 = torch.split(Q0, 1, dim=-1)
    r1, i1, j1, k1 = torch.split(Q1, 1, dim=-1)

    res = torch.cat([
        r0*r1 - i0*i1 - j0*j1 - k0*k1,
        r0*i1 + i0*r1 + j0*k1 - k0*j1,
        r0*j1 - i0*k1 + j0*r1 + k0*i1,
        r0*k1 + i0*j1 - j0*i1 + k0*r1
    ], dim=-1)

    return res
    
def Q_inv(Q):
    """
    Args:
        Q: (..., 4)
    """
    if Q.shape[-1] != 4:
        raise ValueError(f"Invalid quaternion shape {Q.shape}")

    res = torch.tensor([1, -1, -1, -1], dtype=torch.float32, device=Q.device) * Q
    return res

""" Operations with R6 """
def R6_to_R(R6):
    """
    Parameters
        R6: (..., 6)
    """
    if R6.shape[-1] != 6:
        raise ValueError(f"Invalid R6 shape {R6.shape}")
    
    x_, y_ = R6[..., :3], R6[..., 3:]
    x = F.normalize(x_, dim=-1)
    y = F.normalize(y_ - (x * y_).sum(dim=-1, keepdim=True) * x, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], dim=-2) # (..., 3, 3)