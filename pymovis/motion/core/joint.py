import numpy as np

from pymovis.motion.utils import npconst

class Joint:
    def __init__(
        self,
        name  :str,
        offset:np.ndarray=npconst.P_ZERO()
    ):
        self.name = name
        self.offset = offset