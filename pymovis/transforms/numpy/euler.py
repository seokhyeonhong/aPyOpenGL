import numpy as np

from . import quat

def to_rotmat(angles, order, radians=True):
    if not radians:
        angles = np.deg2rad(angles)

    def _euler_axis_to_rotmat(angle, axis):
        one  = np.ones_like(angle, dtype=np.float32)
        zero = np.zeros_like(angle, dtype=np.float32)
        cos  = np.cos(angle, dtype=np.float32)
        sin  = np.sin(angle, dtype=np.float32)

        if axis == "x":
            rotmat_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "y":
            rotmat_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "z":
            rotmat_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        else:
            raise ValueError(f"Invalid axis: {axis}")
        return np.stack(rotmat_flat, axis=-1).reshape(angle.shape + (3, 3))

    Rs = [_euler_axis_to_rotmat(angles[..., i], order[i]) for i in range(3)]
    return np.matmul(np.matmul(Rs[0], Rs[1]), Rs[2])

def to_quat(angles, order, radians=True):
    if not radians:
        angles = np.deg2rad(angles)
    
    def _euler_axis_to_quat(angle, axis):
        zero = np.zeros_like(angle, dtype=np.float32)
        cos  = np.cos(angle / 2, dtype=np.float32)
        sin  = np.sin(angle / 2, dtype=np.float32)

        if axis == "x":
            quat_flat = (cos, sin, zero, zero)
        elif axis == "y":
            quat_flat = (cos, zero, sin, zero)
        elif axis == "z":
            quat_flat = (cos, zero, zero, sin)
        else:
            raise ValueError(f"Invalid axis: {axis}")
        return np.stack(quat_flat, axis=-1).reshape(angle.shape + (4,))
    
    qs = [_euler_axis_to_quat(angles[..., i], order[i]) for i in range(3)]
    return quat.mul(quat.mul(qs[0], qs[1]), qs[2])
