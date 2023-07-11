from __future__ import annotations

import numpy as np
import copy

from .joint import Joint
from pymovis.utils import npconst
from pymovis.ops import mathops, rotation

class Skeleton:
    """
    Hierarchical structure of joints

    Attributes:
        joints      (list[Joint]): List of joints
        v_up        (np.ndarray): Up vector of the skeleton
        v_forward   (np.ndarray): Forward vector of the skeleton
        parent_id   (list[int]): List of parent ids
        children_id (list[list[int]]): List of children ids
        id_by_name  (dict[str, int]): Dictionary of joint ids by name
    """
    def __init__(
        self,
        joints: list[Joint] = None,
    ):
        self.__joints: list[Joint]           = [] if joints is None else joints
        self.__parent_idx: list[int]         = []
        self.__children_idx: list[list[int]] = []
        self.__idx_by_name: dict             = {}
        self.__pre_xforms: np.ndarray        = None
    
    # get functions
    def get_joints(self):
        return self.__joints
    
    def get_parent_idx(self):
        return self.__parent_idx
    
    def get_parent_idx_of(self, idx):
        return self.__parent_idx[idx]
    
    def get_children_idx(self):
        return self.__children_idx
    
    def get_idx_by_name(self, name):
        idx = self.__idx_by_name.get(name, None)
        if name is None:
            raise ValueError(f"Name {name} does not exist.")
        return idx
    
    # set functions
    def set_pre_Q_of(self, idx, pre_Q):
        self.__joints[idx].set_pre_Q(pre_Q)

    def set_local_p_of(self, idx, local_p):
        self.__joints[idx].set_local_p(local_p)

    def num_joints(self):
        return len(self.__joints)

    def add_joint(self, joint_name, pre_Q=None, local_p=None, parent_idx=None):
        # add parent and children indices
        if parent_idx is None or parent_idx == -1:
            if len(self.__joints) > 0:
                raise ValueError(f"Root joint {self.__joints[0].get_name()} already exists. Cannot add {joint_name}.")
            self.__parent_idx.append(-1)
        else:
            self.__parent_idx.append(parent_idx)
            self.__children_idx[parent_idx].append(len(self.__joints))
        
        # add joint
        joint = Joint(joint_name, pre_Q, local_p)
        self.__children_idx.append(list())
        self.__idx_by_name[joint_name] = len(self.__joints)
        self.__joints.append(joint)
    
    def get_pre_xforms(self):
        if self.__pre_xforms is None:
            self.__pre_xforms = np.stack([joint.get_pre_xform() for joint in self.__joints])
        return self.__pre_xforms