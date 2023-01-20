import torch
import torch.nn.functional as F

from pymovis.motion.core import Skeleton
from pymovis.utils import torchconst

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

To avoid confusion, we use __ to denote the function argument if it has the same name as a class.
For example, to denote the rotation matrix, we use __R instead of R.
"""

def normalize_vector(x, dim=-1, eps=1e-6):
    return F.normalize(x, p=2, dim=dim, eps=eps)

""" FK """
def R_fk(R, root_p, skeleton):
        """
        Parameters
            R: (..., N, 3, 3)
            root_p: (..., 3)
            bone_offset: (N, 3)
            parents: (N,)
        """
        bone_offsets = torch.from_numpy(skeleton.get_bone_offsets()).to(R.device)
        parents = skeleton.parent_idx

        global_R, global_p = [R[..., 0, :, :]], [root_p]
        for i in range(1, len(parents)):
            global_R.append(torch.matmul(global_R[parents[i]], R[..., i, :, :]))
            global_p.append(torch.matmul(global_R[parents[i]], bone_offsets[i]) + global_p[parents[i]])
        
        global_R = torch.stack(global_R, dim=-3) # (..., N, 3, 3)
        global_p = torch.stack(global_p, dim=-2) # (..., N, 3)
        return global_R, global_p


def R6_fk(R6: torch.Tensor, root_p: torch.Tensor, skeleton: Skeleton):
    """
    Parameters
        R6: (..., N, 6)
        root_p: (..., 3)
        bone_offset: (N, 3)
        parents: (N,)
    """
    if R6.shape[-1] != 6:
        raise ValueError(f"R6.shape[-1] = {R6.shape[-1]} != 6")
    R = R6_to_R(R6)
    R, p = R_fk(R, root_p, skeleton)
    return R_to_R6(R), p

""" Conversion from R """
def R_to_R6(R: torch.Tensor) -> torch.Tensor:
    """
    Parameters
        R: (..., 3, 3)
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"r.shape[-2:] = {R.shape[-2:]} != (3, 3)")
    x = normalize_vector(R[..., 0, :], axis=-1)
    y = normalize_vector(R[..., 1, :], axis=-1)
    return torch.cat([x, y], dim=-1) # (..., 6)

""" Conversion from E """
def E_to_R(E: torch.Tensor, order: str, radians: bool=True) -> torch.Tensor:
    """
    Parameters
        E: (..., 3)
    """
    if not radians:
        E = torch.deg2rad(E)

    R_map = {
        "x": lambda x: torch.stack([torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x),
                                    torch.zeros_like(x), torch.cos(x), -torch.sin(x),
                                    torch.zeros_like(x), torch.sin(x), torch.cos(x)], dim=-1).reshape(*x.shape, 3, 3),
        "y": lambda y: torch.stack([torch.cos(y), torch.zeros_like(y), torch.sin(y),
                                    torch.zeros_like(y), torch.ones_like(y), torch.zeros_like(y),
                                    -torch.sin(y), torch.zeros_like(y), torch.cos(y)], dim=-1).reshape(*y.shape, 3, 3),
        "z": lambda z: torch.stack([torch.cos(z), -torch.sin(z), torch.zeros_like(z),
                                    torch.sin(z), torch.cos(z), torch.zeros_like(z),
                                    torch.zeros_like(z), torch.zeros_like(z), torch.ones_like(z)], dim=-1).reshape(*z.shape, 3, 3)
    }

    if len(order) == 3:
        R0 = R_map[order[0]](E[..., 0])
        R1 = R_map[order[1]](E[..., 1])
        R2 = R_map[order[2]](E[..., 2])
        return torch.matmul(R0, torch.matmul(R1, R2))
    elif len(order) == 1:
        return R_map[order](E)
    else:
        raise ValueError(f"Invalid order: {order}")


def E_to_Q(E: torch.Tensor, order: str) -> torch.Tensor:
    axis = {
        'x': torch.tensor([1, 0, 0], dtype=torch.float32, device=E.device),
        'y': torch.tensor([0, 1, 0], dtype=torch.float32, device=E.device),
        'z': torch.tensor([0, 0, 1], dtype=torch.float32, device=E.device)
    }

    q0 = A_to_Q(E[..., 0], axis[order[0]])
    q1 = A_to_Q(E[..., 1], axis[order[1]])
    q2 = A_to_Q(E[..., 2], axis[order[2]])

    return Q_mul(q0, Q_mul(q1, q2))

""" Conversion from A """
def A_to_R(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """
    Parameters
        angle: (..., N)
        axis:  (..., 3)
    """
    if axis.shape[-1] != 3:
        raise ValueError(f"axis.shape[-1] = {axis.shape[-1]} != 3")
    
    if angle.shape == axis.shape[:-1]:
        angle = angle[..., None] # (..., N, 1)

    a0, a1, a2     = axis[..., 0], axis[..., 1], axis[..., 2]
    zero           = torch.zeros_like(a0)
    skew_symmetric = torch.stack([zero, -a2, a1,
                                a2, zero, -a0,
                                -a1, a0, zero], dim=-1).reshape(*angle.shape[:-1], 1, 3, 3) # (..., 1, 3, 3)
    I              = torch.eye(3, dtype=torch.float32, device=angle.device)              # (3, 3)
    I              = torch.tile(I, reps=[*angle.shape[:-1], 1, 1])[..., None, :, :]      # (..., 1, 3, 3)
    sin            = torch.sin(angle)[..., None, None]                                   # (..., N, 1, 1)
    cos            = torch.cos(angle)[..., None, None]                                   # (..., N, 1, 1)
    return I + skew_symmetric * sin + torch.matmul(skew_symmetric, skew_symmetric) * (1 - cos) # (..., N, 3, 3)

def A_to_Q(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """
    Parameters
        angle: angles tensor (..., N)
        axis: axis tensor (..., 3)
    """
    if axis.shape[-1] != 3:
        raise ValueError(f"axis.shape[-1] = {axis.shape[-1]} != 3")

    axis = normalize_vector(axis, dim=-1)
    a0, a1, a2 = axis[..., 0], axis[..., 1], axis[..., 2]
    cos = torch.cos(angle / 2)[..., None]
    sin = torch.sin(angle / 2)[..., None]

    return torch.cat([cos, a0 * sin, a1 * sin, a2 * sin], dim=-1) # (..., 4)

""" Conversion from Q """
def Q_to_R(Q: torch.Tensor) -> torch.Tensor:
    """
    Parameters
        Q: (..., 4)
    """
    if Q.shape[-1] != 4:
        raise ValueError(f"q.shape[-1] = {Q.shape[-1]} != 4")
    
    Q = normalize_vector(Q, dim=-1)
    w, x, y, z = torch.unbind(Q, dim=-1)

    row0 = torch.stack([2*(w*w + x*x) - 1, 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1) # (..., 3)
    row1 = torch.stack([2*(w*z + x*y), 2*(w*w + y*y) - 1, 2*(y*z - w*x)], dim=-1) # (..., 3)
    row2 = torch.stack([2*(x*z - w*y), 2*(w*x + y*z), 2*(w*w + z*z) - 1], dim=-1) # (..., 3)
    return torch.stack([row0, row1, row2], dim=-2) # (..., 3, 3)

def Q_to_R6(Q: torch.Tensor) -> torch.Tensor:
    """
    :param __Q: (..., 4)
    """
    if Q.shape[-1] != 4:
        raise ValueError(f"q.shape[-1] = {Q.shape[-1]} != 4")
    
    Q = normalize_vector(Q, dim=-1)
    w, x, y, z = torch.unbind(Q, dim=-1)

    r0 = torch.stack([2*(w*w + x*x) - 1, 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1)
    r1 = torch.stack([2*(x*y + w*z), 2*(w*w + y*y) - 1, 2*(y*z - w*x)], dim=-1)
    return torch.cat([r0, r1], dim=-1) # (..., 6)

""" Conversion from R6 """
def R6_to_R(R6: torch.Tensor) -> torch.Tensor:
    """
    Parameters
        R6: (..., 6)
    """
    if R6.shape[-1] != 6:
        raise ValueError(f"R6.shape[-1] = {R6.shape[-1]} != 6")
    
    x = normalize_vector(R6[..., 0:3]) # (..., 3)
    y = normalize_vector(R6[..., 3:6] - torch.sum(x * R6[..., 3:6], dim=-1, keepdim=True) * x) # (..., 3)
    z = torch.cross(x, y, dim=-1) # (..., 3)
    return torch.stack([x, y, z], dim=-2) # (..., 3, 3)

""" Operations for R """
def R_inv(R: torch.Tensor) -> torch.Tensor:
        """
        Parameters
            R: (..., N, 3, 3)
        """
        if R.shape[-2:] != (3, 3):
            raise ValueError(f"r.shape[-2:] = {R.shape[-2:]} != (3, 3)")
        return R.transpose(-1, -2)

def R_mul(R0: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
    """
    Parameters
        R0: (..., N, 3, 3)
        R1: (..., N, 3, 3)
    """
    if R0.shape[-2:] != (3, 3):
        raise ValueError(f"R0.shape[-2:] = {R0.shape[-2:]} != (3, 3)")
    if R1.shape[-2:] != (3, 3):
        raise ValueError(f"R1.shape[-2:] = {R1.shape[-2:]} != (3, 3)")
    return torch.matmul(R0, R1)

""" Operations for Q """
def Q_mul(Q0: torch.Tensor, Q1: torch.Tensor) -> torch.Tensor:
    """
    :param q0: left-sided quaternion (..., 4)
    :param q1: right-sided quaternion (..., 4)
    :return: q0 * q1 (..., 4)
    """
    if Q0.shape[-1] != 4 or Q1.shape[-1] != 4:
        raise ValueError(f"q0.shape[-1] = {Q0.shape[-1]} != 4 or q1.shape[-1] = {Q1.shape[-1]} != 4")
    w0, x0, y0, z0 = torch.unbind(Q0, dim=-1)
    w1, x1, y1, z1 = torch.unbind(Q1, dim=-1)

    res = torch.cat([
        w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
        w1 * x0 + x1 * w0 - y1 * z0 + z1 * y0,
        w1 * y0 + x1 * z0 + y1 * w0 - z1 * x0,
        w1 * z0 - x1 * y0 + y1 * x0 + z1 * w0], dim=-1)

    return res
    
def Q_inv(Q: torch.Tensor) -> torch.Tensor:
    """
    :param q: quaternion tensor (..., 4)
    :return: inverse quaternion tensor (..., 4)
    """
    if Q.shape[-1] != 4:
        raise ValueError(f"q.shape[-1] = {Q.shape[-1]} != 4")

    res = torch.tensor([1, -1, -1, -1], dtype=torch.float32, device=Q.device) * Q
    return res