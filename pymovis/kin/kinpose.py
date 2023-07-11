import numpy as np

from ..vis import Pose

class KinPose:
    def __init__(self):
        self._basis_xform = np.eye(4, dtype=np.float32)
    
    def recompute_local_root(self):
        basis_inv = np.linalg.inv(self._basis_xform)