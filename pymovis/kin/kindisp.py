import numpy as np

from .kinpose import KinPose

class KinDisp:
    def __init__(self, source: KinPose, target: KinPose):
        self.source = source
        self.target = target

        # delta
        self.d_basis_xform = 
        self.d_local_Rs = 
        self.d_local_root_p = 