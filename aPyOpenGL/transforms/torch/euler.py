import torch

from . import quat

def to_rotmat(angles, order, radians=True):
    if not radians:
        angles = torch.deg2rad(angles)

    def _euler_axis_to_rotmat(angle, axis):
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
    
    Rs = [_euler_axis_to_rotmat(angles[..., i], order[i]) for i in range(3)]
    return torch.matmul(torch.matmul(Rs[0], Rs[1]), Rs[2])

def to_quat(angles, order, radians=True):
    if not radians:
        angles = torch.deg2rad(angles)
    
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
        
    qs = [_euler_axis_to_Q(angles[..., i], order[i]) for i in range(3)]
    return quat.mul(quat.mul(qs[0], qs[1]), qs[2])
