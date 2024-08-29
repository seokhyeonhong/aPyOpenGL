from __future__ import annotations
import numpy as np
import copy

from .pose import Pose
from aPyOpenGL.transforms import n_quat

class Motion:
    """
    Motion class that contains the skeleton and its sequence of poses.

    Attributes:
        poses     (list[Pose]): A sequence of poses.
        fps       (float)     : The number of frames per second.
        name      (str)       : The name of the motion.
    """
    def __init__(
        self,
        poses : list[Pose],
        fps   : float = 30.0,
        name  : str   = "default",
    ):
        self.poses : list[Pose] = poses
        self.fps   : float      = fps
        self.name  : str        = name

    def __len__(self):
        return len(self.poses)
    
    @property
    def num_frames(self):
        return len(self.poses)
    
    @property
    def skeleton(self):
        return self.poses[0].skeleton
    
    def remove_joint_by_name(self, joint_name):
        remove_indices = self.skeleton.remove_joint_by_name(joint_name)
        for pose in self.poses:
            pose.skeleton = self.skeleton
            pose.local_quats = np.delete(pose.local_quats, remove_indices, axis=0)
    
    def mirror(self, pair_indices):
        mirrored_poses = []
        for pose in self.poses:
            mirrored_poses.append(pose.mirror(pair_indices))
        return Motion(mirrored_poses, self.fps, self.name)