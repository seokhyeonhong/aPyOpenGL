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
        name: str = None,
        pre_Q: np.ndarray = None,
        local_p: np.ndarray = None
    ):
        self.__name = None if name is None else str(name)
        self.__pre_Q = None if pre_Q is None else np.array(pre_Q, dtype=np.float32)
        self.__local_p = None if local_p is None else np.array(local_p, dtype=np.float32)
    
    # get functions
    def get_name(self):
        if self.__name is None:
            raise ValueError("Name is not initialized.")
        return self.__name
    
    def get_pre_Q(self):
        if self.__pre_Q is None:
            raise ValueError("Pre-rotation quaternion is not initialized.")
        return self.__pre_Q
    
    def get_local_p(self):
        if self.__local_p is None:
            raise ValueError("Local position is not initialized.")
        return self.__local_p
    
    # set functions
    def set_name(self, name: str):
        if self.__name is not None:
            raise ValueError("Name already initialized.")
        self.__name = str(name)

    def set_pre_Q(self, pre_Q: np.ndarray):
        if self.__pre_Q is not None:
            raise ValueError("Pre-rotation quaternion already initialized.")
        self.__pre_Q = np.array(pre_Q, dtype=np.float32)

    def set_local_p(self, local_p: np.ndarray):
        if self.__local_p is not None:
            raise ValueError("Local position already initialized.")
        self.__local_p = np.array(local_p, dtype=np.float32)

    def is_initialized(self):
        return all([
            self.__name is not None,
            self.__pre_Q is not None,
            self.__local_p is not None
        ])

    def get_pre_xform(self):
        if not self.is_initialized():
            raise ValueError("Joint is not initialized.")
        
        pre_xform = np.eye(4, dtype=np.float32)
        pre_xform[:3, :3] = rotation.Q_to_R(self.__pre_Q)
        pre_xform[:3, 3] = self.__local_p
        return np.ascontiguousarray(pre_xform)