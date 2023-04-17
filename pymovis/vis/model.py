import copy
from OpenGL.GL import *

from pymovis.vis.mesh import Mesh
from pymovis.vis.const import LAFAN1_FBX_DICT

class Model:
    def __init__(self, gl_meshes=None, skeleton=None):
        self.gl_meshes = gl_meshes
        self.skeleton = skeleton
        if gl_meshes is not None:
            self.meshes = [Mesh(gl_meshes[i][0], gl_meshes[i][1], skeleton=skeleton) for i in range(len(gl_meshes))]
        else:
            raise ValueError("Mesh must be provided")
    
    def __deepcopy__(self, memo):
        meshes = [copy.deepcopy(mesh) for mesh in self.meshes]
        res = Model(self.gl_meshes, self.skeleton)
        res.meshes = meshes
        memo[id(self)] = res
        return res

    def set_pose_by_source(self, source_pose):
        for mesh in self.meshes:
            mesh.set_pose_by_source(source_pose)
    
    def set_source_skeleton(self, source, rel_dict=LAFAN1_FBX_DICT):
        for mesh in self.meshes:
            mesh.set_source_skeleton(source, rel_dict)