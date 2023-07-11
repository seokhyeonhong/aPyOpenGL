import numpy as np

"""
NumPy implementation of vector and rotation operations.
Each class is a wrapper around a NumPy array.
The value of the array is stored in the `value` attribute, and its shape can be multiple dimensions.
For example, the shape of NumpyVec3 can be not only (3,) but also (3, 3) or (..., 3).
"""

class NumpyVec3:
    def __init__(self, value=None):
        self.value = np.array(value, dtype=np.float32) if value is not None else np.zeros(3, dtype=np.float32)
        if self.value.shape[-1] != 3:
            raise ValueError(f"Vec3 must be 3-dimensional, but got shape {self.value.shape}")
        
    def __str__(self): return f"Vec3({self.value})"
    def __repr__(self): return f"Vec3({self.value})"

    def __add__(self, other: "NumpyVec3"): return NumpyVec3(self.value + other.value)
    def __sub__(self, other: "NumpyVec3"): return NumpyVec3(self.value - other.value)
    def __mul__(self, other: "NumpyVec3"): return NumpyVec3(self.value * other.value)
    def __truediv__(self, other: "NumpyVec3"): return NumpyVec3(self.value / other.value)
    def __neg__(self): return NumpyVec3(-self.value)
    def __pos__(self): return NumpyVec3(+self.value)

class NumpyVec4:
    def __init__(self, value=None):
        self.value = np.array(value, dtype=np.float32) if value is not None else np.array([0, 0, 0, 1], dtype=np.float32)
        if self.value.shape[-1] != 4:
            raise ValueError(f"Vec4 must be 4-dimensional, but got shape {self.value.shape}")
        
    def __str__(self): return f"Vec4({self.value})"
    def __repr__(self): return f"Vec4({self.value})"

    def __add__(self, other: "NumpyVec4"): return NumpyVec4(self.value + other.value)
    def __sub__(self, other: "NumpyVec4"): return NumpyVec4(self.value - other.value)
    def __mul__(self, other: "NumpyVec4"): return NumpyVec4(self.value * other.value)
    def __truediv__(self, other: "NumpyVec4"): return NumpyVec4(self.value / other.value)
    def __neg__(self): return NumpyVec4(-self.value)
    def __pos__(self): return NumpyVec4(+self.value)

class NumpyQuat:
    