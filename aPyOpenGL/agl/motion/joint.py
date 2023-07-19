from __future__ import annotations

import numpy as np
from aPyOpenGL.transforms import n_quat, n_rotmat

class Joint:
    """
    Joint of a skeleton.

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
        self.__pre_quat = np.array([1, 0, 0, 0], dtype=np.float32) if pre_quat is None else np.array(pre_quat, dtype=np.float32)
        self.__local_pos = np.array([0, 0, 0], dtype=np.float32) if local_pos is None else np.array(local_pos, dtype=np.float32)

        if self.__pre_quat.shape != (4,):
            raise ValueError(f"Pre-rotation quaternion must be a 4-dimensional vector, but got {self.__pre_quat.shape}.")
        if self.__local_pos.shape != (3,):
            raise ValueError(f"Local position must be a 3-dimensional vector, but got {self.__local_pos.shape}.")
        
        self._recompute_pre_xform()

    @property
    def pre_quat(self):
        return self.__pre_quat
    
    @property
    def local_pos(self):
        return self.__local_pos
    
    @pre_quat.setter
    def pre_quat(self, value):
        self.__pre_quat = np.array(value, dtype=np.float32)
        if self.__pre_quat.shape != (4,):
            raise ValueError(f"Pre-rotation quaternion must be a 4-dimensional vector, but got {self.__pre_quat.shape}.")
        self._recompute_pre_xform()
    
    @local_pos.setter
    def local_pos(self, value):
        self.__local_pos = np.array(value, dtype=np.float32)
        if self.__local_pos.shape != (3,):
            raise ValueError(f"Local position must be a 3-dimensional vector, but got {self.__local_pos.shape}.")
        self._recompute_pre_xform()

    def _recompute_pre_xform(self):
        pre_rotmat = n_quat.to_rotmat(self.__pre_quat)
        self.pre_xform = n_rotmat.to_xform(pre_rotmat, translation=self.__local_pos)