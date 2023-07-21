from __future__ import annotations
import numpy as np

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