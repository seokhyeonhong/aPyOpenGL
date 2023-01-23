from OpenGL.GL import *

# from pymovis.motion.core import Skeleton
from pymovis.vis.mesh import Mesh

class Model:
    def __init__(self, gl_meshes=None, skeleton=None):
        if gl_meshes is not None and skeleton is not None:
            self.meshes = []
            for i in range(len(gl_meshes)):
                self.meshes.append(Mesh(gl_meshes[i][0], gl_meshes[i][1], skeleton=skeleton))
        
        elif gl_meshes is not None:
            # create skeleton
            # skeleton = Skeleton()
            # skeleton.add_joint("root")

            # set meshes
            self.meshes = []
            for i in range(len(gl_meshes)):
                self.meshes.append(Mesh(gl_meshes[i][0], gl_meshes[i][1]))
        else:
            raise ValueError("Mesh must be provided")

    def set_pose_by_source(self, source_pose):
        for mesh in self.meshes:
            mesh.set_pose_by_source(source_pose)
    
    def set_source_skeleton(self, source, rel_dict):
        for mesh in self.meshes:
            mesh.set_source_skeleton(source, rel_dict)