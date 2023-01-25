from OpenGL.GL import *
import glm
import numpy as np

# from pymovis.motion.core import Pose
from pymovis.vis.core import MeshGL

class Mesh:
    def __init__(
        self,
        mesh_gl: MeshGL,
        materials=None,
        use_skinning=False,
        skeleton=None
    ):
        self.mesh_gl      = mesh_gl
        self.materials    = materials
        self.use_skinning = use_skinning or skeleton is not None
        self.skeleton     = skeleton
        self.buffer       = [glm.mat4(1.0)] * len(self.mesh_gl.joint_order)
        
    def set_materials(self, materials):
        self.materials = materials
    
    def set_pose_by_source(self, pose):
        if self.skeleton is None:
            return
        
        buffer_updated = [False] * len(self.mesh_gl.joint_order)
        self.buffer = [glm.mat4(1.0)] * len(self.mesh_gl.joint_order)

        global_R, global_p = pose.global_R, pose.global_p
        for i in range(self.source_skeleton.num_joints):
            world_trf = np.concatenate((global_R[i], global_p[i][:, None]), axis=1)
            world_trf = np.concatenate((world_trf, np.array([[0, 0, 0, 1]])), axis=0).astype(np.float32)
            world_trf = glm.mat4(*world_trf.T.ravel())
            
            source_joint_name = self.source_skeleton.joints[i].name
            target_joint_name = self.rel_dict[source_joint_name]
            target_joint_idx  = self.mesh_gl.name_to_idx[target_joint_name]

            bind_trf_inv = self.mesh_gl.joint_bind_trf_inv[target_joint_idx]
            self.buffer[target_joint_idx] = world_trf * bind_trf_inv
            buffer_updated[target_joint_idx] = True

        for i, updated in enumerate(buffer_updated):
            if not updated:
                name = self.mesh_gl.joint_order[i]
                self.buffer[i] = self.buffer[self.mesh_gl.name_to_idx[name]-1]

    def set_source_skeleton(self, source, rel_dict):
        if self.skeleton is None:
            return
            
        self.source_skeleton = source
        self.rel_dict = rel_dict