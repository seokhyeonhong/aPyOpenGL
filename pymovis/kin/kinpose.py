import numpy as np

from ..agl import Pose
from pymovis.transforms import n_quat

class KinPose:
    """
    Represents a pose of a skeleton with a basis transformation.
    It doesn't modify the original pose data.
    
    global_xforms
        - root: basis_xform @ pre_xforms[0] @ local_xforms[0]
        - others: global_xforms[parent] @ pre_xforms[i] @ local_xforms[i]

    basis_xform (4, 4)
        - 4x4 transformation matrix in global space
        - initialized by the root joint

    local_xforms (J, 4, 4)
        - root: relative to the basis
        - others: relative to the parent
    """
    def __init__(
        self,
        pose: Pose
    ):
        # original pose data - NO CHANGE
        self.pose = pose
        self.skeleton = pose.skeleton
        self.root_pre_R = n_quat.to_rotmat(self.skeleton.joints[0].pre_quat)

        self._recompute_local_root()

    def _recompute_local_root(self):
        # local transformations
        # - root: relative to the basis
        # - others: relative to the parent
        self.local_root_p = self.pose.root_pos
        self.local_Rs = n_quat.to_rotmat(self.pose.local_quats)

        # basis transformation (4, 4)
        self.basis_xform = self.get_projected_root_xform()

        # root transformation relative to the basis
        root_xform = self.pose.global_xforms()[0]
        self.local_Rs[0] = np.linalg.inv(self.root_pre_R) @ np.linalg.inv(self.basis_xform[:3, :3]) @ root_xform[:3, :3]
        self.local_root_p = (np.linalg.inv(self.basis_xform) @ root_xform)[:3, 3]

    def get_projected_root_xform(self):
        basis_xform = np.eye(4, dtype=np.float32)

        # get root transformation from the original pose
        root_xform = self.pose.global_xforms()[0]
        root_R = root_xform[:3, :3]
        root_p = root_xform[:3, 3]
        
        # set rotation by column vectors
        dir_x = root_R[:, 0] * np.array([1, 0, 1], dtype=np.float32)
        dir_x = dir_x / (np.linalg.norm(dir_x) + 1e-8)

        dir_z = root_R[:, 2] * np.array([1, 0, 1], dtype=np.float32)
        dir_z = dir_z / (np.linalg.norm(dir_z) + 1e-8)

        dir_y = np.array([0, 1, 0], dtype=np.float32)

        basis_xform[:3, :3] = np.stack([dir_x, dir_y, dir_z], axis=1)

        # set position
        basis_xform[:3, 3] = root_p * np.array([1, 0, 1], dtype=np.float32)

        return basis_xform

    def set_basis_xform(self, xform):
        self.basis_xform = np.array(xform, dtype=np.float32)
        if self.basis_xform.shape != (4, 4):
            raise ValueError(f"basis_xform must be 4x4, not {self.basis_xform.shape}")
    
    def transform_basis(self, delta):
        self.basis_xform = delta @ self.basis_xform

    def set_pose(self, pose: Pose):
        self.pose = pose
        self._recompute_local_root()
        
    def to_pose(self) -> Pose:
        local_Rs = self.local_Rs.copy()

        # recompute local root transformation
        # - rotation: global_R = basis_R @ pre_R @ local_R_to_basis = pre_R @ local_R_for_pose
        #             Therefore, local_R_for_pose = inv(pre_R) @ basis_R @ pre_R @ local_R_to_basis
        # - position: global_p = basis_R @ local_pos + basis_p
        local_Rs[0] = np.linalg.inv(self.root_pre_R) @ self.basis_xform[:3, :3] @ self.root_pre_R @ local_Rs[0]
        root_p = self.basis_xform @ np.concatenate([self.local_root_p, [1]])
        return Pose(self.skeleton, n_quat.from_rotmat(local_Rs), root_p[:3])