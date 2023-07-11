from OpenGL.GL import *
import glm
import copy

from .motion import Skeleton, Pose
from .core   import MeshGL

class Mesh:
    def __init__(
        self,
        mesh_gl: MeshGL,
        materials    = None,
        skeleton: Skeleton = None
    ):
        self.mesh_gl      = mesh_gl
        self.materials    = materials

        # skinning
        self.skeleton     = skeleton
        self.use_skinning = (skeleton is not None)
        self.buffer       = [glm.mat4(1.0)] * len(self.mesh_gl.joint_names)
        
    def __deepcopy__(self, memo):
        res = Mesh(self.mesh_gl, copy.deepcopy(self.materials), self.use_skinning, self.skeleton)
        res.buffer = copy.deepcopy(self.buffer)
        memo[id(self)] = res
        return res

    def set_materials(self, materials):
        self.materials = materials
    
    def update_mesh(self, pose: Pose):
        if self.skeleton is None:
            return
        
        self.buffer = [glm.mat4(1.0)] * len(self.mesh_gl.joint_names)
        for i in range(len(self.mesh_gl.joint_names)):
            jidx = self.skeleton.get_idx_by_name(self.mesh_gl.joint_names[i])
            global_xform = glm.mat4(*pose.get_global_xforms()[jidx].T.ravel())
            bind_xform_inv = self.mesh_gl.joint_bind_xform_inv[i]
            self.buffer[i] = global_xform * bind_xform_inv