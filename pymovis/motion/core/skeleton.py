from __future__ import annotations

import numpy as np

from pymovis.motion.core.joint import Joint
from pymovis.motion.utils import npconst

class Skeleton:
    """
    Defines the hierarchical structure of joints.
    """
    def __init__(
        self,
        joints   : list[Joint]=[],
        v_up     : np.ndarray=npconst.UP(),
        v_forward: np.ndarray=npconst.FORWARD(),
    ):
        assert v_up.shape == (3,), f"v_up.shape = {v_up.shape}"
        assert v_forward.shape == (3,), f"v_forward.shape = {v_forward.shape}"

        self.joints = joints
        self.v_up = v_up
        self.v_forward = v_forward
        self.parent_id = []
        self.children_id = []
        self.id_by_name = {}
    
    @property
    def num_joints(self):
        return len(self.joints)
    
    def add_joint(self, joint_name, parent_id=None):
        joint_id = len(self.joints)

        if parent_id == None:
            assert len(self.joints) == 0, "Only one root joint is allowed"
            self.parent_id.append(-1)
            self.children_id.append([])
        else:
            self.parent_id.append(parent_id)
            self.children_id[parent_id].append(joint_id)

        joint = Joint(joint_name)
        self.id_by_name[joint_name] = len(self.joints)
        self.joints.append(joint)
        self.children_id.append([])
    
    def get_bone_offsets(self):
        res = [joint.offset for joint in self.joints]
        return np.stack(res, axis=0)
    
    def get_joint_by_name(self, name):
        return self.joints[self.id_by_name[name]]