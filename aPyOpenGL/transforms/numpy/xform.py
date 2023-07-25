import numpy as np

from . import rotmat, quat, aaxis

"""
Operations
"""
def interpolate(x0, x1, t):
    r0, p0 = x0[..., :3, :3], x0[..., :3, 3]
    r1, p1 = x1[..., :3, :3], x1[..., :3, 3]

    r = rotmat.interpolate(r0, r1, t)
    p = p0 + (p1 - p0) * t

    return rotmat.to_xform(r, translation=p)

def fk(local_xforms, root_pos, skeleton):
    """
    Attributes:
        local_xforms: (..., J, 4, 4)
        root_pos: (..., 3), global root position
        skeleton: aPyOpenGL.agl.Skeleton
    """
    pre_xforms = np.tile(skeleton.pre_xforms, local_xforms.shape[:-3] + (1, 1, 1)) # (..., J, 4, 4)
    pre_xforms[..., 0, :3, 3] = root_pos
    
    global_xforms = [pre_xforms[..., 0, :, :] @ local_xforms[..., 0, :, :]]
    for i in range(1, skeleton.num_joints):
        global_xforms.append(global_xforms[skeleton.parent_idx[i]] @ pre_xforms[..., i, :, :] @ local_xforms[..., i, :, :])
    
    global_xforms = np.stack(global_xforms, axis=-3) # (..., J, 4, 4)
    return global_xforms

"""
Transformation matrix to other representation
"""
def to_rotmat(xform):
    return xform[..., :3, :3]

def to_quat(xform):
    return rotmat.to_quat(to_rotmat(xform))

def to_aaxis(xform):
    return quat.to_aaxis(to_quat(xform))

def to_ortho6d(xform):
    return rotmat.to_ortho6d(to_rotmat(xform))

def to_translation(xform):
    return np.ascontiguousarray(xform[..., :3, 3])

"""
Other representation to transformation matrix
"""
def from_rotmat(r, translation=None):
    return rotmat.to_xform(r, translation=translation)

def from_quat(q, translation=None):
    return quat.to_xform(q, translation=translation)

def from_aaxis(a, translation=None):
    return aaxis.to_xform(a, translation=translation)

def from_ortho6d(r, translation=None):
    return rotmat.to_xform(r, translation=translation)