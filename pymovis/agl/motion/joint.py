from __future__ import annotations

import numpy as np
from pymovis.transforms import n_quat, n_rotmat

class Joint:
    """
    Joint of a skeleton.
    name, pre_Q, and local_p must be initialized.

    Attributes:
        name      (str):        Name of the joint
        pre_quat  (np.ndarray): Pre-rotation of the joint in quaternion
        local_pos (np.ndarray): Local position of the joint relative to its parent
    """
    def __init__(
        self,
        name: str,
        pre_quat: np.ndarray = None,
        local_pos: np.ndarray = None
    ):
        self.name = str(name)
        self.pre_quat = np.array([1, 0, 0, 0], dtype=np.float32) if pre_quat is None else np.array(pre_quat, dtype=np.float32)
        self.local_p = np.array([0, 0, 0], dtype=np.float32) if local_pos is None else np.array(local_pos, dtype=np.float32)

        if self.pre_quat.shape != (4,):
            raise ValueError(f"Pre-rotation quaternion must be a 4-dimensional vector, but got {self.pre_quat.shape}.")
        if self.local_p.shape != (3,):
            raise ValueError(f"Local position must be a 3-dimensional vector, but got {self.local_p.shape}.")

    def pre_xform(self):
        pre_rotmat = n_quat.to_rotmat(self.pre_quat)
        return n_rotmat.to_xform(pre_rotmat, translation=self.local_p)