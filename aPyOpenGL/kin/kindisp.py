import numpy as np

from .kinpose import KinPose
from aPyOpenGL.transforms import n_quat, n_xform, n_rotmat

class KinDisp:
    def __init__(self, source: KinPose, target: KinPose):
        self.source = source
        self.target = target

        # delta
        self.d_basis_xform = self.target.basis_xform @ np.linalg.inv(self.source.basis_xform)
        self.d_local_quats = n_quat.mul(self.target.local_quats, n_quat.inv(self.source.local_quats))
        self.d_local_root_pos = self.target.local_root_pos - self.source.local_root_pos
    
    def apply(self, kpose: KinPose):
        # apply delta to the input KinPose
        kpose.basis_xform = self.d_basis_xform @ kpose.basis_xform
        kpose.local_quats = n_quat.mul(self.d_local_quats, kpose.local_quats)
        kpose.local_root_pos = self.d_local_root_pos + kpose.local_root_pos