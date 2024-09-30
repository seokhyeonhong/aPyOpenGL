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

        if len(self.joints) > 0:
            self.recompute_pre_xform()
    
    @property
    def num_joints(self):
        return len(self.joints)

    @property
    def pre_xforms(self):
        return self.__pre_xforms.copy()
    
    def add_joint(self, joint_name, pre_quat=None, local_pos=None, parent_idx=None):
        # add parent and children indices
        if parent_idx is None or parent_idx == -1:
            if len(self.joints) > 0:
                raise ValueError(f"Root joint {self.joints[0].name} already exists. Cannot add {joint_name}.")
            self.parent_idx.append(-1)
        else:
            self.parent_idx.append(parent_idx)
            self.children_idx[parent_idx].append(len(self.joints))
        
        # add joint
        joint = Joint(joint_name, pre_quat, local_pos)
        self.children_idx.append(list())
        self.idx_by_name[joint_name] = len(self.joints)
        self.joints.append(joint)

        # recompute pre-transform
        self.recompute_pre_xform()
    
    def remove_joint_by_name(self, joint_name):
        joint_idx = self.idx_by_name.get(joint_name, None)
        if joint_idx is None:
            raise ValueError(f"Joint {joint_name} does not exist.")
        return self.remove_joint_by_idx(joint_idx)

    def remove_joint_by_idx(self, joint_idx):
        remove_indices = [joint_idx]
        
        def dfs(jidx):
            for cidx in self.children_idx[jidx]:
                remove_indices.append(cidx)
                dfs(cidx)

        dfs(joint_idx)
        remove_indices.sort(reverse=True)

        for ridx in remove_indices:
            self.children_idx[self.parent_idx[ridx]].remove(ridx)
            self.joints.pop(ridx)
            self.parent_idx.pop(ridx)
            self.children_idx.pop(ridx)

        for i in range(len(self.joints)):
            self.parent_idx[i] = self.parent_idx[i] - sum([1 for ridx in remove_indices if ridx < self.parent_idx[i]])
            self.children_idx[i] = [cidx - sum([1 for ridx in remove_indices if ridx < cidx]) for cidx in self.children_idx[i]]
            
        for i, joint in enumerate(self.joints):
            self.idx_by_name[joint.name] = i

        self.recompute_pre_xform()

        return remove_indices

    def recompute_pre_xform(self):
        self.__pre_xforms = np.stack([joint.pre_xform for joint in self.joints])
    
    def find_symmetry_axis(self, pair_indices):
        assert len(self.joints) == len(pair_indices), f"number of pair indices {len(pair_indices)} must be same with the number of joints {len(self.joints)}"

        offsets = self.pre_xforms[:, :3, -1].copy()
        offsets = offsets - offsets[pair_indices]

        x = np.max(np.abs(offsets[:, 0]))
        y = np.max(np.abs(offsets[:, 1]))
        z = np.max(np.abs(offsets[:, 2]))
        
        if x > y and x > z:
            axis = "x"
        elif y > x and y > z:
            axis = "y"
        elif z > x and z > y:
            axis = "z"
        else:
            raise Exception("Symmetry axis not found")
        
        return axis