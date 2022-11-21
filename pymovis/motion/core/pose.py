import numpy as np

from pymovis.motion.core.skeleton import Skeleton
from pymovis.motion.ops.npmotion import R
from pymovis.motion.utils import npconst

from pymovis.vis.render import Render
from pymovis.vis.primitives import Sphere

class Pose:
    """
    Represents a pose of a skeleton.
    It contains the local rotation matrices and the root position.

    Attributes:
        skeleton (Skeleton): The skeleton that this pose belongs to.
        local_R  (numpy.ndarray): The local rotation matrices of the joints.
        root_p   (numpy.ndarray): The root position.
    """
    def __init__(
        self,
        skeleton: Skeleton,
        local_R: np.ndarray,
        root_p: np.ndarray=npconst.P_ZERO(),
    ):
        assert local_R.shape == (skeleton.num_joints, 3, 3), f"local_R.shape = {local_R.shape}"
        assert root_p.shape == (3,), f"root_p.shape = {root_p.shape}"

        self.skeleton = skeleton
        self.local_R = local_R
        self.root_p = root_p
    
    @classmethod
    def from_bvh(cls, skeleton, local_E, order, root_p):
        local_R = R.from_E(local_E, order, radians=False)
        return cls(skeleton, local_R, root_p)
    
    @classmethod
    def from_numpy(cls, skeleton, local_R, root_p):
        return cls(skeleton, local_R, root_p)

    @property
    def forward(self):
        return self.local_R[0] @ self.skeleton.v_forward
    
    @property
    def up(self):
        return self.local_R[0] @ self.skeleton.v_up
    
    @property
    def left(self):
        return np.cross(self.up, self.forward)
    
    def draw(self):
        if not hasattr(self, "joint_sphere"):
            self.joint_sphere = Sphere(0.07)

        _, global_p = R.fk(self.local_R, self.root_p, self.skeleton.get_bone_offsets(), self.skeleton.parent_id)
        for i in range(self.skeleton.num_joints):
            Render.render_options(self.joint_sphere).set_position(global_p[i]).draw()