import numpy as np
import glm

from pymovis.motion.core.skeleton import Skeleton
from pymovis.motion.ops.npmotion import R
from pymovis.motion.utils import npconst

from pymovis.vis.render import Render
from pymovis.vis.primitives import Sphere, Cylinder

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

    @classmethod
    def from_torch(cls, skeleton, local_R, root_p):
        return cls(skeleton, local_R.cpu().numpy(), root_p.cpu().numpy())

    @property
    def forward(self):
        return self.local_R[0] @ self.skeleton.v_forward
    
    @property
    def up(self):
        return self.local_R[0] @ self.skeleton.v_up
    
    @property
    def left(self):
        return np.cross(self.up, self.forward)
    
    def draw(self, albedo=glm.vec3(1.0, 0.0, 0.0)):
        if not hasattr(self, "joint_sphere"):
            self.joint_sphere = Sphere(0.05)
            self.joint_bone   = Cylinder(0.03, 1.0)

        _, global_p = R.fk(self.local_R, self.root_p, self.skeleton)
        for i in range(self.skeleton.num_joints):
            Render.render_options(self.joint_sphere)\
                .set_position(global_p[i])\
                .set_material(albedo=albedo)\
                .draw()

            if i != 0:
                parent_pos = global_p[self.skeleton.parent_id[i]]

                center = glm.vec3((parent_pos + global_p[i]) / 2)
                dist = np.linalg.norm(parent_pos - global_p[i])
                dir = glm.vec3((global_p[i] - parent_pos) / dist)

                axis = glm.cross(glm.vec3(0, 1, 0), dir)
                angle = glm.acos(glm.dot(glm.vec3(0, 1, 0), dir))
                rotation = glm.rotate(glm.mat4(1.0), angle, axis)
                
                Render.render_options(self.joint_bone)\
                    .set_position(center)\
                    .set_orientation(rotation)\
                    .set_scale(glm.vec3(1.0, dist, 1.0))\
                    .set_material(albedo=albedo)\
                    .draw()