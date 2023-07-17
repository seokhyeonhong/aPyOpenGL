from __future__ import annotations

import numpy as np

from .joint import Joint

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
        self.joints: list[Joint]           = [] if joints is None else joints
        self.parent_idx: list[int]         = []
        self.children_idx: list[list[int]] = []
        self.idx_by_name: dict             = {}
    
    def num_joints(self):
        return len(self.joints)

    def add_joint(self, joint_name, pre_Q=None, local_p=None, parent_idx=None):
        # add parent and children indices
        if parent_idx is None or parent_idx == -1:
            if len(self.joints) > 0:
                raise ValueError(f"Root joint {self.joints[0].name} already exists. Cannot add {joint_name}.")
            self.parent_idx.append(-1)
        else:
            self.parent_idx.append(parent_idx)
            self.children_idx[parent_idx].append(len(self.joints))
        
        # add joint
        joint = Joint(joint_name, pre_Q, local_p)
        self.children_idx.append(list())
        self.idx_by_name[joint_name] = len(self.joints)
        self.joints.append(joint)
    
    def pre_xforms(self):
        xforms = np.stack([joint.pre_xform() for joint in self.joints])
        return xforms
    
    def root_pre_xform(self):
        return self.joints[0].pre_xform()