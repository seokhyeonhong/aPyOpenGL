import numpy as np

"""
Functions for constant tensors
"""

def X():
    return np.array([1, 0, 0], dtype=np.float32)

def Y():
    return np.array([0, 1, 0], dtype=np.float32)

def Z():
    return np.array([0, 0, 1], dtype=np.float32)

def XY():
    return np.array([1, 1, 0], dtype=np.float32)

def XZ():
    return np.array([1, 0, 1], dtype=np.float32)

def YZ():
    return np.array([0, 1, 1], dtype=np.float32)

def FORWARD():
    return np.array([0, 0, 1], dtype=np.float32)

def UP():
    return np.array([0, 1, 0], dtype=np.float32)

def LEFT():
    return np.array([1, 0, 0], dtype=np.float32)

def P_ZERO():
    return np.zeros(3, dtype=np.float32)