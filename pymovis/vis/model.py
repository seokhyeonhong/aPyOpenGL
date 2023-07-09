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
        skeleton: Skeleton = None
    ):
        if meshes is None and skeleton is None:
            raise ValueError("Both meshes and skeleton cannot be None")
        
        self.pose = Pose(skeleton) if skeleton is not None else None
        self.meshes = [Mesh(meshes[i][0], meshes[i][1], skeleton=skeleton) for i in range(len(meshes))] if meshes is not None else []
        
    def set_pose(self, pose):
        if isinstance(pose, Pose):
            self.pose = pose
        # elif isinstance(pose, KinPose):
        else:
            raise ValueError(f"pose must be Pose or KinPose, not {type(pose)}")
        for mesh in self.meshes:
            mesh.update_mesh(pose)