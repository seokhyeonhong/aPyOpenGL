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

def normalize(x, dim=-1, eps=1e-6):
    return F.normalize(x, p=2, dim=dim, eps=eps)

class R:
    @staticmethod
    def from_A(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        """
        :param angle: (..., N)
        :param axis:  (..., 3)
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

    @staticmethod
    def from_E(__E: torch.Tensor, order: str, radians: bool=True) -> torch.Tensor:
        """
        :param __E: (..., 3)
        """
        if __E.shape[-1] != 3:
            raise ValueError(f"e.shape[-1] = {__E.shape[-1]} != 3")
        
        if not radians:
            __E = torch.deg2rad(__E)

        axis_map = {
            "x": torchconst.X(),
            "y": torchconst.Y(),
            "z": torchconst.Z(),
        }

        R0 = R.from_A(__E[..., 0], axis=axis_map[order[0]])
        R1 = R.from_A(__E[..., 1], axis=axis_map[order[1]])
        R2 = R.from_A(__E[..., 2], axis=axis_map[order[2]])
        return torch.matmul(R0, torch.matmul(R1, R2))
    
    @staticmethod
    def from_Q(__Q: torch.Tensor) -> torch.Tensor:
        """
        :param __Q: (..., 4)
        """
        if __Q.shape[-1] != 4:
            raise ValueError(f"q.shape[-1] = {__Q.shape[-1]} != 4")
        
        __Q = normalize(__Q, dim=-1)
        w, x, y, z = torch.unbind(__Q, dim=-1)

        row0 = torch.stack([2*(w*w + x*x) - 1, 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1) # (..., 3)
        row1 = torch.stack([2*(w*z + x*y), 2*(w*w + y*y) - 1, 2*(y*z - w*x)], dim=-1) # (..., 3)
        row2 = torch.stack([2*(x*z - w*y), 2*(w*x + y*z), 2*(w*w + z*z) - 1], dim=-1) # (..., 3)
        return torch.stack([row0, row1, row2], dim=-2) # (..., 3, 3)

    @staticmethod
    def from_R6(__R6: torch.Tensor) -> torch.Tensor:
        """
        :param __R6: (..., 6)
        """
        if __R6.shape[-1] != 6:
            raise ValueError(f"__R6.shape[-1] = {__R6.shape[-1]} != 6")
        
        x = normalize(__R6[..., 0:3]) # (..., 3)
        y = normalize(__R6[..., 3:6] - torch.sum(x * __R6[..., 3:6], dim=-1, keepdim=True) * x) # (..., 3)
        z = torch.cross(x, y, dim=-1) # (..., 3)
        return torch.stack([x, y, z], dim=-2) # (..., 3, 3)

    @staticmethod
    def inv(__R: torch.Tensor) -> torch.Tensor:
        """
        :param __R: (..., N, 3, 3)
        """
        if __R.shape[-2:] != (3, 3):
            raise ValueError(f"r.shape[-2:] = {__R.shape[-2:]} != (3, 3)")
        return __R.transpose(-1, -2)
    
    @staticmethod
    def mul(R0: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
        """
        :param R0: (..., N, 3, 3)
        :param R1: (..., N, 3, 3)
        """
        if R0.shape[-2:] != (3, 3):
            raise ValueError(f"R0.shape[-2:] = {R0.shape[-2:]} != (3, 3)")
        if R1.shape[-2:] != (3, 3):
            raise ValueError(f"R1.shape[-2:] = {R1.shape[-2:]} != (3, 3)")
        return torch.matmul(R0, R1)

    @staticmethod
    def fk(__R: torch.Tensor, root_p: torch.Tensor, skeleton: Skeleton):
        """
        :param __R: (..., N, 3, 3)
        :param root_p: (..., 3)
        :param bone_offset: (N, 3)
        :param parents: (N,)
        """
        bone_offsets = torch.from_numpy(skeleton.get_bone_offsets()).to(__R.device)
        parents = skeleton.parent_idx
        global_R, global_p = [__R[..., 0, :, :]], [root_p]
        for i in range(1, len(parents)):
            global_R.append(torch.matmul(global_R[parents[i]], __R[..., i, :, :]))
            global_p.append(torch.matmul(global_R[parents[i]], bone_offsets[i]) + global_p[parents[i]])
        
        global_R = torch.stack(global_R, dim=-3) # (..., N, 3, 3)
        global_p = torch.stack(global_p, dim=-2) # (..., N, 3)
        return global_R, global_p

    @staticmethod
    def local_p(global_p: torch.Tensor, local_R: torch.Tensor) -> torch.Tensor:
        """
        :param global_p: (..., N, 3)
        :param local_R: (..., N, 3, 3)
        """
        if global_p.shape[-1] != 3:
            raise ValueError(f"global_p.shape[-1] = {global_p.shape[-1]} != 3")
        if local_R.shape[-2:] != (3, 3):
            raise ValueError(f"root_R.shape[-2:] = {local_R.shape[-2:]} != (3, 3)")

        p = global_p - global_p[..., 0:1, :] # (..., N, 3)
        root_R = local_R[..., 0, :, :]       # (..., 3, 3)
        return torch.matmul(R.inv(root_R), p.transpose(-1, -2)).transpose(-1, -2).contiguous()
    
class R6:
    @staticmethod
    def from_Q(__Q: torch.Tensor) -> torch.Tensor:
        """
        :param __Q: (..., 4)
        """
        if __Q.shape[-1] != 4:
            raise ValueError(f"q.shape[-1] = {__Q.shape[-1]} != 4")
        
        __Q = normalize(__Q, dim=-1)
        w, x, y, z = __Q[..., 0], __Q[..., 1], __Q[..., 2], __Q[..., 3]

        r0 = torch.stack([2*(w*w + x*x) - 1, 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1)
        r1 = torch.stack([2*(x*y + w*z), 2*(w*w + y*y) - 1, 2*(y*z - w*x)], dim=-1)
        return torch.cat([r0, r1], dim=-1) # (..., 6)

    @staticmethod
    def from_R(__R: torch.Tensor) -> torch.Tensor:
        """
        Assumes that rows of '__R' are unit vectors.
        :param __R: (..., 3, 3)
        """
        if __R.shape[-2:] != (3, 3):
            raise ValueError(f"r.shape[-2:] = {__R.shape[-2:]} != (3, 3)")
        x = __R[..., 0, :]
        y = __R[..., 1, :]
        return torch.cat([x, y], dim=-1) # (..., 6)

    @staticmethod
    def fk(__R6: torch.Tensor, root_p: torch.Tensor, skeleton: Skeleton):
        """
        :param __R6: (..., N, 6)
        :param root_p: (..., 3)
        :param bone_offset: (N, 3)
        :param parents: (N,)
        """
        if __R6.shape[-1] != 6:
            raise ValueError(f"R6.shape[-1] = {__R6.shape[-1]} != 6")
        r = R.from_R6(__R6)
        r, p = R.fk(r, root_p, skeleton)
        return R6.from_R(r), p
    
    @staticmethod
    def mul(R0: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
        """
        :param R0: (..., 6)
        :param R1: (..., 6)
        """
        if R0.shape[-1] != 6:
            raise ValueError(f"R0.shape[-1] = {R0.shape[-1]} != 6")
        if R1.shape[-1] != 6:
            raise ValueError(f"R1.shape[-1] = {R1.shape[-1]} != 6")
        r0 = R.from_R6(R0)
        r1 = R.from_R6(R1)
        r = R.mul(r0, r1)
        return R6.from_R(r)

class Q:
    @staticmethod
    def from_A(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        """
        :param angle: angles tensor (..., N)
        :param axis: axis tensor (..., 3)
        :return: quaternion tensor
        """
        if axis.shape[-1] != 3:
            raise ValueError(f"axis.shape[-1] = {axis.shape[-1]} != 3")

        axis = normalize(axis, dim=-1)
        a0, a1, a2 = axis[..., 0], axis[..., 1], axis[..., 2]
        cos = torch.cos(angle / 2)[..., None]
        sin = torch.sin(angle / 2)[..., None]

        return torch.cat([cos, a0 * sin, a1 * sin, a2 * sin], dim=-1) # (..., 4)

    @staticmethod
    def from_E(__E: torch.Tensor, order: str) -> torch.Tensor:
        axis = {
            'x': torch.tensor([1, 0, 0], dtype=torch.float32, device=__E.device),
            'y': torch.tensor([0, 1, 0], dtype=torch.float32, device=__E.device),
            'z': torch.tensor([0, 0, 1], dtype=torch.float32, device=__E.device)
        }

        q0 = Q.from_A(__E[..., 0], axis[order[0]])
        q1 = Q.from_A(__E[..., 1], axis[order[1]])
        q2 = Q.from_A(__E[..., 2], axis[order[2]])

        return Q.mul(q0, Q.mul(q1, q2))
    
    @staticmethod
    def mul(Q0: torch.Tensor, Q1: torch.Tensor) -> torch.Tensor:
        """
        :param q0: left-sided quaternion (..., 4)
        :param q1: right-sided quaternion (..., 4)
        :return: q0 * q1 (..., 4)
        """
        if Q0.shape[-1] != 4 or Q1.shape[-1] != 4:
            raise ValueError(f"q0.shape[-1] = {Q0.shape[-1]} != 4 or q1.shape[-1] = {Q1.shape[-1]} != 4")
        w0, x0, y0, z0 = Q0[..., 0:1], Q0[..., 1:2], Q0[..., 2:3], Q0[..., 3:4]
        w1, x1, y1, z1 = Q1[..., 0:1], Q1[..., 1:2], Q1[..., 2:3], Q1[..., 3:4]

        res = torch.cat([
            w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
            w1 * x0 + x1 * w0 - y1 * z0 + z1 * y0,
            w1 * y0 + x1 * z0 + y1 * w0 - z1 * x0,
            w1 * z0 - x1 * y0 + y1 * x0 + z1 * w0], dim=-1)

        return res
    
    @staticmethod
    def inv(__Q: torch.Tensor) -> torch.Tensor:
        """
        :param q: quaternion tensor (..., 4)
        :return: inverse quaternion tensor (..., 4)
        """
        if __Q.shape[-1] != 4:
            raise ValueError(f"q.shape[-1] = {__Q.shape[-1]} != 4")

        res = torch.tensor([1, -1, -1, -1], dtype=torch.float32, device=__Q.device) * __Q
        return res