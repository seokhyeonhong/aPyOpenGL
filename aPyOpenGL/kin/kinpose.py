import numpy as np

from ..agl import Pose
from ..transforms import n_quat, n_xform

class KinPose:
    """
    Represents a pose of a skeleton with a basis transformation.
    It doesn't modify the original pose data.

    NOTE: We assume that the pre-rotation is defined so that the forward direction of the root joint is the z-axis in the world coordinate when the local root rotation is identity.
    
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
    def __init__(self, pose: Pose):
        # original pose data - NO CHANGE
        self.pose = pose
        self.skeleton = pose.skeleton
        self.root_pre_quat = self.skeleton.joints[0].pre_quat

        self._recompute_local_root()

    def _recompute_local_root(self):
        # local transformations
        # - root: relative to the basis
        # - others: relative to the parent
        self.local_root_pos = self.pose.root_pos.copy()
        self.local_quats = self.pose.local_quats.copy()

        # basis transformation (4, 4)
        self.basis_xform = self.get_projected_root_xform()

        # root transformation relative to the basis
        root_xform = n_quat.to_xform(self.pose.local_quats[0], self.pose.root_pos)
        self.local_quats[0] = n_quat.mul(n_quat.inv(self.root_pre_quat), n_quat.from_rotmat(self.basis_xform[:3, :3].T @ root_xform[:3, :3])) # root_pre_rot.inv() * basis_rot.inv() * root_rot
        self.local_root_pos = self.basis_xform[:3, :3].T @ (self.pose.root_pos - self.basis_xform[:3, 3])

    def get_projected_root_xform(self):
        # basis: world forward -> root forward
        root_fwd = n_quat.mul_vec(self.pose.local_quats[0], np.array([0, 0, 1], dtype=np.float32))
        root_fwd = root_fwd * np.array([1, 0, 1], dtype=np.float32)
        root_fwd = root_fwd / (np.linalg.norm(root_fwd) + 1e-8)
        
        world_fwd = np.array([0, 0, 1], dtype=np.float32)

        basis_quat = n_quat.between_vecs(world_fwd, root_fwd)

        # basis
        basis_rotmat = n_quat.to_rotmat(basis_quat)
        basis_pos    = self.pose.root_pos * np.array([1, 0, 1], dtype=np.float32)
        basis_xform  = n_xform.from_rotmat(basis_rotmat, basis_pos)

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
        local_quats = self.local_quats.copy()

        # recompute local "root" transformation
        # - rotation: global_rot = basis_rot * pre_rot * local_rot_to_basis = pre_rot * local_rot_for_pose
        #             Therefore, local_rot_for_pose = pre_rot.inv() * basis_rot * pre_rot * local_rot_to_basis
        # - position: global_pos = basis_rot * local_pos + basis_pos
        q0 = n_quat.mul(n_quat.inv(self.root_pre_quat), n_quat.from_rotmat(self.basis_xform[:3, :3])) # pre_rot.inv() * basis_rot
        q1 = n_quat.mul(self.root_pre_quat, local_quats[0]) # pre_rot * local_rot_to_basis
        local_quats[0] = n_quat.mul(q0, q1)
        root_pos = self.basis_xform @ np.concatenate([self.local_root_pos, [1]])
        return Pose(self.skeleton, local_quats, root_pos[:3])