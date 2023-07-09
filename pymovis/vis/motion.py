from __future__ import annotations

import numpy as np
import copy

from pymovis.utils import npconst
from pymovis.ops import mathops, rotation

class Joint:
    """
    Joint of a skeleton

    Attributes:
        name    (str):        Name of the joint
        pre_Q   (np.ndarray): Pre-rotation of the joint in quaternion
        local_p (np.ndarray): Local position of the joint relative to its parent
    """
    def __init__(
        self,
        name: str = "joint",
        pre_Q: np.ndarray = npconst.Q_IDENTITY(),
        local_p: np.ndarray = npconst.P_ZERO()
    ):
        self.name    = name
        self.pre_Q   = np.array(pre_Q, dtype=np.float32)
        self.local_p = np.array(local_p, dtype=np.float32)
        if self.pre_Q.shape[-1] != 4:
            raise ValueError(f"Invalid shape of pre-rotation quaternion: {self.pre_Q.shape}")
        if self.local_p.shape[-1] != 3:
            raise ValueError(f"Invalid shape of local position: {self.local_p.shape}")
    
    def get_pre_xform(self):
        pre_xform = np.eye(4, dtype=np.float32)
        pre_xform[:3, :3] = rotation.Q_to_R(self.pre_Q)
        pre_xform[:3, 3] = self.local_p
        return pre_xform

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
    
    @property
    def num_joints(self):
        return len(self.joints)
    
    @property
    def effector_idx(self):
        res = []
        for i in range(len(self.joints)):
            if len(self.children_idx[i]) == 0:
                res.append(i)
        return res

    def add_joint(self, joint_name, local_p=npconst.P_ZERO(), pre_Q=npconst.Q_IDENTITY(), parent_idx=None):
        # add parent and children indices
        if parent_idx is None or parent_idx == -1:
            if len(self.joints) > 0:
                raise ValueError(f"Root joint {self.joints[0].name} already exists. Cannot add {joint_name}.")
            self.parent_idx.append(-1)
        else:
            self.parent_idx.append(parent_idx)
            self.children_idx[parent_idx].append(len(self.joints))
            
        self.children_idx.append(list())

        # add joint
        joint = Joint(joint_name, pre_Q=pre_Q, local_p=local_p)
        self.idx_by_name[joint_name] = len(self.joints)
        self.joints.append(joint)
    
    def get_pre_xforms(self):
        pre_xforms = np.stack([joint.get_pre_xform() for joint in self.joints], axis=0)
        return pre_xforms

class Pose:
    """
    Represents a pose of a skeleton.
    It contains the local rotation matrices of each joint and the root position.

    Attributes:
        skeleton (Skeleton)     : The skeleton that this pose belongs to.
        local_R  (numpy.ndarray): The local rotation matrices of the joints.
        root_p   (numpy.ndarray): The root position.
    """
    def __init__(
        self,
        skeleton: Skeleton,
        local_Qs: np.ndarray or list[np.ndarray] = None,
        root_p  : np.ndarray = None,
    ):
        self.skeleton = skeleton
        self.local_Qs = np.array(local_Qs, dtype=np.float32) if local_Qs is not None else np.stack([npconst.Q_IDENTITY() for _ in range(skeleton.num_joints)], axis=0)
        self.root_p   = np.array(root_p, dtype=np.float32) if root_p is not None else skeleton.joints[0].local_p

        # check shapes
        if self.skeleton.num_joints == 0:
            raise ValueError("Cannot create a pose for an empty skeleton.")
        if self.local_Qs.shape != (skeleton.num_joints, 4):
            raise ValueError(f"local_R.shape must be ({skeleton.num_joints}, 4), but got {local_Qs.shape}")
        if self.root_p.shape != (3,):
            raise ValueError(f"root_p.shape must be (3,), but got {root_p.shape}")
        
        # transformations
        self.local_xforms    = self._get_local_xforms()
        self.global_xforms   = self._get_global_xforms()
        self.skeleton_xforms = self._get_skeleton_xforms()

    def _get_local_xforms(self):
        local_Rs = rotation.Q_to_R(self.local_Qs)
        local_xforms = np.stack([np.identity(4, dtype=np.float32) for _ in range(self.skeleton.num_joints)], axis=0)
        local_xforms[0, :3, :3] = local_Rs[0]
        local_xforms[0, :3, 3]  = self.root_p
        for i in range(1, self.skeleton.num_joints):
            local_xforms[i, :3, :3] = local_Rs[i]
        
        return local_xforms
    
    def _get_global_xforms(self):
        pre_xforms = self.skeleton.get_pre_xforms()
        global_xforms = np.stack([np.identity(4, dtype=np.float32) for _ in range(self.skeleton.num_joints)], axis=0)
        global_xforms[0] = self.local_xforms[0]
        for i in range(1, self.skeleton.num_joints):
            global_xforms[i] = global_xforms[self.skeleton.parent_idx[i]] @ pre_xforms[i] @ self.local_xforms[i]
        
        return global_xforms
    
    def _get_skeleton_xforms(self):
        skeleton_xforms = np.stack([np.identity(4, dtype=np.float32) for _ in range(self.skeleton.num_joints - 1)], axis=0)
        for i in range(1, self.skeleton.num_joints):
            parent_pos = self.global_xforms[self.skeleton.parent_idx[i], :3, 3]
            
            target_dir = mathops.normalize(self.global_xforms[i, :3, 3] - parent_pos)
            axis = mathops.normalize(np.cross(npconst.UP(), target_dir))
            angle = mathops.signed_angle(npconst.UP(), target_dir, axis)

            skeleton_xforms[i-1, :3, :3] = rotation.A_to_R(angle, axis)
            skeleton_xforms[i-1, :3,  3] = (parent_pos + self.global_xforms[i, :3, 3]) / 2
        
        return skeleton_xforms

    @classmethod
    def from_bvh(cls, skeleton, local_E, order, root_p):
        local_Qs = rotation.E_to_Q(local_E, order, radians=False)
        return cls(skeleton, local_Qs, root_p)
    
    @classmethod
    def from_numpy(cls, skeleton, local_R, root_p):
        return cls(skeleton, local_R, root_p)

    @classmethod
    def from_torch(cls, skeleton, local_R, root_p):
        return cls(skeleton, local_R.cpu().numpy(), root_p.cpu().numpy())

    # """ Base position and directions (on xz plane, equivalent to horizontal plane) """
    # @property
    # def base(self):
    #     return self.root_p * npconst.XZ()
    
    # @property
    # def forward(self):
    #     return mathops.normalize((self.local_R[0] @ self.skeleton.v_forward) * npconst.XZ())
    
    # @property
    # def up(self):
    #     return npconst.UP()
    
    # @property
    # def left(self):
    #     return mathops.normalize(np.cross(self.up, self.forward))

    # """ Manipulation functions """
    # def set_root_p(self, root_p):
    #     delta = root_p - self.root_p
    #     self.translate_root_p(delta)

    # def translate_root_p(self, delta):
    #     self.root_p += delta
    #     self.global_p += delta
    
    # def rotate_root(self, delta):
    #     self.local_R[0] = np.matmul(delta, self.local_R[0])
    #     self.global_R, self.global_p = motionops.R_fk(self.local_R, self.root_p, self.skeleton)
    
    # def update(self):
    #     """ Called whenever the pose is modified """
    #     self.global_R, self.global_p = motionops.R_fk(self.local_R, self.root_p, self.skeleton)

    # """ IK functions """
    # def two_bone_ik(self, base_idx, effector_idx, target_p, eps=1e-8, facing="forward"):
    #     mid_idx = self.skeleton.parent_idx[effector_idx]
    #     if self.skeleton.parent_idx[mid_idx] != base_idx:
    #         raise ValueError(f"{base_idx} and {effector_idx} are not in a two bone IK hierarchy")

    #     a = self.global_p[base_idx]
    #     b = self.global_p[mid_idx]
    #     c = self.global_p[effector_idx]

    #     global_a_R = self.global_R[base_idx]
    #     global_b_R = self.global_R[mid_idx]

    #     lab = np.linalg.norm(b - a)
    #     lcb = np.linalg.norm(b - c)
    #     lat = np.clip(np.linalg.norm(target_p - a), eps, lab + lcb - eps)

    #     ac_ab_0 = np.arccos(np.clip(np.dot(mathops.normalize(c - a), mathops.normalize(b - a)), -1, 1))
    #     ba_bc_0 = np.arccos(np.clip(np.dot(mathops.normalize(a - b), mathops.normalize(c - b)), -1, 1))
    #     ac_at_0 = np.arccos(np.clip(np.dot(mathops.normalize(c - a), mathops.normalize(target_p - a)), -1, 1))

    #     ac_ab_1 = np.arccos(np.clip((lcb*lcb - lab*lab - lat*lat) / (-2*lab*lat), -1, 1))
    #     ba_bc_1 = np.arccos(np.clip((lat*lat - lab*lab - lcb*lcb) / (-2*lab*lcb), -1, 1))

    #     axis_0 = mathops.normalize(np.cross(c - a, self.forward if facing == "forward" else -self.forward))
    #     axis_1 = mathops.normalize(np.cross(c - a, target_p - a))

    #     r0 = rotation.A_to_R(ac_ab_1 - ac_ab_0, rotation.R_inv(global_a_R) @ axis_0)
    #     r1 = rotation.A_to_R(ba_bc_1 - ba_bc_0, rotation.R_inv(global_b_R) @ axis_0)
    #     r2 = rotation.A_to_R(ac_at_0, rotation.R_inv(global_a_R) @ axis_1)

    #     self.local_R[base_idx] = self.local_R[base_idx] @ r0 @ r2
    #     self.local_R[mid_idx] = self.local_R[mid_idx] @ r1

    #     self.update()

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

    @classmethod
    def from_numpy(cls, skeleton, local_R, root_p, fps=30.0, name="default", type="default"):
        poses = [Pose.from_numpy(skeleton, local_R[i], root_p[i]) for i in range(local_R.shape[0])]
        return cls(skeleton, poses, fps=fps, name=name, type=type)

    @classmethod
    def from_torch(cls, skeleton, local_R, root_p, fps=30.0, name="default", type="default"):
        poses = [Pose.from_numpy(skeleton, local_R[i].cpu().numpy(), root_p[i].cpu().numpy()) for i in range(local_R.shape[0])]
        return cls(skeleton, poses, fps=fps, name=name, type=type)

    def make_window(self, start, end):
        return Motion(
            copy.deepcopy(self.poses[start:end]),
            self.fps,
            self.name,
        )
    
    def copy(self):
        return copy.deepcopy(self)

    """ Alignment functions """
    def align_to_origin_by_frame(self, frame, axes="xyz"):
        delta = -self.poses[frame].root_p
        if "x" not in axes:
            delta[0] = 0
        if "y" not in axes:
            delta[1] = 0
        if "z" not in axes:
            delta[2] = 0
            
        for pose in self.poses:
            pose.translate_root_p(delta)
    
    def align_to_forward_by_frame(self, frame, forward=npconst.FORWARD()):
        forward_from = self.poses[frame].forward
        forward_to   = mathops.normalize(forward * npconst.XZ())

        angle = mathops.signed_angle(forward_from, forward_to)
        axis = np.array([0, 1, 0], dtype=np.float32)
        R_delta = rotation.A_to_R(angle, axis)
        
        base = self.poses[frame].base
        for pose in self.poses:
            pose.local_R[0] = np.matmul(R_delta, pose.local_R[0])
            pose.root_p = np.matmul(R_delta, (pose.root_p - base).T).T + base
            pose.update()
    
    def align_by_frame(self, frame, origin_axes="xyz", forward=npconst.FORWARD()):
        """
        Args:
            frame (int) : The frame to align to.
            origin_axes (str) : The axes to align the origin to. "x", "y", "z", or any combination of them.
            forward (np.array) : The forward direction to align to.
        """
        self.align_to_origin_by_frame(frame, origin_axes)
        self.align_to_forward_by_frame(frame, forward)