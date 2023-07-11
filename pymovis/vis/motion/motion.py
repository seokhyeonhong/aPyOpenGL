from __future__ import annotations

import numpy as np
import copy

from .pose import Pose
from pymovis.utils import npconst
from pymovis.ops import mathops, rotation

class Motion:
    """
    Motion class that contains the skeleton and its sequence of poses.

    Attributes:
        name      (str)       : The name of the motion.
        skeleton  (Skeleton)  : The skeleton that this motion belongs to.
        poses     (list[Pose]): The sequence of poses.
        fps       (float)     : The number of frames per second.
        frametime (float)     : The time between two frames.
    """
    def __init__(
        self,
        poses : list[Pose],
        fps   : float = 30.0,
        name  : str   = "default",
    ):
        self.__poses : list[Pose] = poses
        self.__fps   : float      = fps
        self.__name  : str        = name

    def __len__(self):
        return len(self.__poses)
        
    def num_frames(self):
        return len(self.__poses)
    
    def get_poses(self):
        return self.__poses
    
    def get_pose_at(self, frame):
        return self.__poses[frame]
    
    def get_fps(self):
        return self.__fps
    
    def get_frametime(self):
        return 1.0 / self.__fps
    
    def get_name(self):
        return self.__name

    # @classmethod
    # def from_numpy(cls, skeleton, local_R, root_p, fps=30.0, name="default", type="default"):
    #     poses = [Pose.from_numpy(skeleton, local_R[i], root_p[i]) for i in range(local_R.shape[0])]
    #     return cls(skeleton, poses, fps=fps, name=name, type=type)

    # @classmethod
    # def from_torch(cls, skeleton, local_R, root_p, fps=30.0, name="default", type="default"):
    #     poses = [Pose.from_numpy(skeleton, local_R[i].cpu().numpy(), root_p[i].cpu().numpy()) for i in range(local_R.shape[0])]
    #     return cls(skeleton, poses, fps=fps, name=name, type=type)

    # def make_window(self, start, end):
    #     return Motion(
    #         copy.deepcopy(self.__poses[start:end]),
    #         self.__fps,
    #         self.__name,
    #     )
    
    # def copy(self):
    #     return copy.deepcopy(self)

    # """ Alignment functions """
    # def align_to_origin_by_frame(self, frame, axes="xyz"):
    #     delta = -self.poses[frame].root_p
    #     if "x" not in axes:
    #         delta[0] = 0
    #     if "y" not in axes:
    #         delta[1] = 0
    #     if "z" not in axes:
    #         delta[2] = 0
            
    #     for pose in self.poses:
    #         pose.translate_root_p(delta)
    
    # def align_to_forward_by_frame(self, frame, forward=npconst.FORWARD()):
    #     forward_from = self.poses[frame].forward
    #     forward_to   = mathops.normalize(forward * npconst.XZ())

    #     angle = mathops.signed_angle(forward_from, forward_to)
    #     axis = np.array([0, 1, 0], dtype=np.float32)
    #     R_delta = rotation.A_to_R(angle, axis)
        
    #     base = self.poses[frame].base
    #     for pose in self.poses:
    #         pose.local_R[0] = np.matmul(R_delta, pose.local_R[0])
    #         pose.root_p = np.matmul(R_delta, (pose.root_p - base).T).T + base
    #         pose.update()
    
    # def align_by_frame(self, frame, origin_axes="xyz", forward=npconst.FORWARD()):
    #     """
    #     Args:
    #         frame (int) : The frame to align to.
    #         origin_axes (str) : The axes to align the origin to. "x", "y", "z", or any combination of them.
    #         forward (np.array) : The forward direction to align to.
    #     """
    #     self.align_to_origin_by_frame(frame, origin_axes)
    #     self.align_to_forward_by_frame(frame, forward)