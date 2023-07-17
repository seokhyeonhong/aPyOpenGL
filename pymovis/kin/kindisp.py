import numpy as np

from .kinpose import KinPose

class KinDisp:
    def __init__(self, source: KinPose, target: KinPose):
        self.source = source
        self.target = target

        # delta
        self.d_basis_xform = self.target.basis_xform @ np.linalg.inv(self.source.basis_xform)
        self.d_local_Rs = self.target.local_Rs @ np.linalg.inv(self.source.local_Rs)
        self.d_local_root_p = self.target.local_root_p - self.source.local_root_p
    
    def apply(self, kpose: KinPose):
        # apply delta
        kpose.basis_xform = self.d_basis_xform @ kpose.basis_xform
        kpose.local_Rs = self.d_local_Rs @ kpose.local_Rs
        kpose.local_root_p = self.d_local_root_p + kpose.local_root_p