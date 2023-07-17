from __future__ import annotations

import numpy as np

from pymovis.ops import rotation

class Joint:
    """
    Joint of a skeleton.
    name, pre_Q, and local_p must be initialized.

    Attributes:
        name    (str):        Name of the joint
        pre_Q   (np.ndarray): Pre-rotation of the joint in quaternion
        local_p (np.ndarray): Local position of the joint relative to its parent
    """
    def __init__(
        self,
        name: str,
        pre_Q: np.ndarray = None,
        local_p: np.ndarray = None
    ):
        self.name = str(name)
        self.pre_Q = np.array([1, 0, 0, 0], dtype=np.float32) if pre_Q is None else np.array(pre_Q, dtype=np.float32)
        self.local_p = np.array([0, 0, 0], dtype=np.float32) if local_p is None else np.array(local_p, dtype=np.float32)

        if self.pre_Q.shape != (4,):
            raise ValueError(f"Pre-rotation quaternion must be a 4-dimensional vector, but got {self.pre_Q.shape}.")
        if self.local_p.shape != (3,):
            raise ValueError(f"Local position must be a 3-dimensional vector, but got {self.local_p.shape}.")

    def pre_xform(self):
        pre_R = rotation.Q_to_R(self.pre_Q)
        return rotation.Rp_to_T(pre_R, self.local_p)