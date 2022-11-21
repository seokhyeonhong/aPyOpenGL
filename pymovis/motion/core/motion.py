from __future__ import annotations

import numpy as np
import copy

from pymovis.motion.core.skeleton import Skeleton
from pymovis.motion.core.pose import Pose
from pymovis.motion.ops import npmotion
from pymovis.motion.utils import npconst

class Motion:
    def __init__(
        self,
        name: str,
        skeleton: Skeleton,
        poses: list[Pose],
        global_v: np.ndarray = None,
        fps: float=30.0,
    ):
        self.name = name
        self.skeleton = skeleton
        self.poses = poses
        self.fps = fps
        self.frametime = 1.0 / fps

        # local rotations and root positions are stored separately from the poses
        # to make the computation faster
        self.local_R = np.stack([pose.local_R for pose in poses], axis=0)
        self.root_p  = np.stack([pose.root_p for pose in poses], axis=0)
        if global_v is None:
            _, global_p = npmotion.R.fk(self.local_R, self.root_p, self.skeleton)
            self.global_v = global_p[1:] - global_p[:-1]
            self.global_v = np.pad(self.global_v, ((1, 0), (0, 0), (0, 0)), "edge")
        else:
            self.global_v = global_v
            
    @property
    def num_frames(self):
        return len(self.poses)
    
    @classmethod
    def from_numpy(cls, skeleton, local_R, root_p, fps=30.0):
        poses = []
        for i in range(local_R.shape[0]):
            pose = Pose.from_numpy(skeleton, local_R[i], root_p[i])
            poses.append(pose)
        return cls("motion", skeleton, poses, fps=fps)

    @classmethod
    def from_torch(cls, skeleton, local_R, root_p, fps=30.0):
        poses = []
        for i in range(local_R.shape[0]):
            pose = Pose.from_numpy(skeleton, local_R[i].numpy(), root_p[i].numpy())
            poses.append(pose)
        return cls("motion", skeleton, poses, fps=fps)

    def make_window(self, start, end):
        return Motion(
            self.name,
            self.skeleton,
            copy.deepcopy(self.poses[start:end]),
            copy.deepcopy(self.global_v[start:end]),
            self.fps
        )

    def update(self):
        """
        Called whenever self.local_R or self.root_p are changed.
        """
        for i in range(self.num_frames):
            self.poses[i].local_R = self.local_R[i]
            self.poses[i].root_p = self.root_p[i]
    
    def get_pose_by_frame(self, frame):
        return self.poses[frame]

    def get_pose_by_time(self, time):
        frame = int(time * self.fps)
        return self.poses[frame]

    """
    Alignment functions
    """
    def align_to_origin_by_frame(self, frame):
        self.root_p -= self.root_p[frame] * npconst.XZ()
        self.update()
    
    def align_to_forward_by_frame(self, frame, forward=npconst.FORWARD()):
        forward_from = self.poses[frame].forward
        forward_from = npmotion.normalize(forward_from * npconst.XZ())
        forward_to   = npmotion.normalize(forward * npconst.XZ())

        # if forward_from and forward_to are (nearly) parallel, do nothing
        if np.dot(forward_from, forward_to) > 0.999:
            return
        
        axis = npmotion.normalize(np.cross(forward_from, forward_to))
        angle = np.arccos(np.dot(forward_from, forward_to))
        R_delta = npmotion.R.from_A(angle, axis)
        
        # update root rotation - R: (nof, noj, 3, 3), R_delta: (3, 3)
        self.local_R[:, 0] = np.matmul(R_delta, self.local_R[:, 0])

        # update root position - R_delta: (3, 3), p: (nof, 3) -> (nof, 3)
        root_p_new = self.root_p - self.root_p[frame]
        root_p_new = np.matmul(R_delta, root_p_new.T).T + self.root_p[frame]
        self.root_p[..., (0, 2)] = root_p_new[..., (0, 2)]

        # update velocity - R_delta: (3, 3), v: (nof, noj, 3) -> (nof, noj, 3)
        global_v_new = np.einsum("ij,klj->kli", R_delta, self.global_v)
        self.global_v[..., (0, 2)] = global_v_new[..., (0, 2)]

        self.update()
    
    def align_by_frame(self, frame, forward=npconst.FORWARD()):
        self.align_to_origin_by_frame(frame)
        self.align_to_forward_by_frame(frame, forward)
    
    """
    Rendering
    """
    def render_by_time(self, time):
        frame = int(time * self.fps)
        print(frame)
        self.poses[frame].draw()
    
    def render_by_frame(self, frame):
        self.poses[frame].draw()

    """
    Motion features
    """
    def get_local_R6(self):
        return npmotion.R6.from_R(self.local_R)
    
    def get_root_p(self):
        return self.root_p
    
    def get_root_v(self):
        return self.global_v[:, 0, :]

    def get_contacts(self, lfoot_idx, rfoot_idx, velfactor=0.0002, keep_shape=False):
        """
        Extracts binary tensors of feet contacts

        :param pos: tensor of global positions of shape (Timesteps, Joints, 3)
        :param lfoot_idx: indices list of left foot joints
        :param rfoot_idx: indices list of right foot joints
        :param velfactor: velocity threshold to consider a joint moving or not
        :return: binary tensors of left foot contacts and right foot contacts
        """
        contacts_l = np.linalg.norm(self.global_v[:, lfoot_idx], axis=-1) < velfactor
        contacts_r = np.linalg.norm(self.global_v[:, rfoot_idx], axis=-1) < velfactor

        return np.concatenate([contacts_l, contacts_r], axis=-1, dtype=np.float32)