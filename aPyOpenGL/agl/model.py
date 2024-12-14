from __future__ import annotations
from OpenGL.GL import *

from .core import MeshGL
from .material import Material
from .motion import Skeleton, Pose
from .mesh import Mesh

class Model:
    def __init__(
        self,
        meshes: list[tuple[MeshGL, Material]] = None,
        skeleton: Skeleton = None,
        joint_map: dict[str, str] = None,
    ):
        if meshes is None and skeleton is None:
            raise ValueError("Both meshes and skeleton cannot be None")
        if skeleton is None and joint_map is not None:
            raise ValueError("Joint map requires a skeleton")
        
        self.skeleton = skeleton
        self.meshes = [Mesh(meshes[i][0], meshes[i][1], skeleton=skeleton, joint_map=joint_map) for i in range(len(meshes))] if meshes is not None else []
    
    def set_identity_joint_map(self):
        if self.skeleton is None:
            raise ValueError("Joint map requires a skeleton")
        joint_map = {joint.name: joint.name for joint in self.skeleton.joints}
        self.set_joint_map(joint_map)

    def set_joint_map(self, joint_map: dict[str, str]):
        for mesh in self.meshes:
            mesh.joint_map = joint_map

    def set_pose(self, pose: Pose):
        for mesh in self.meshes:
            mesh.update_mesh(pose)