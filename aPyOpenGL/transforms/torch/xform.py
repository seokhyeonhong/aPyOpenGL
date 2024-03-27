import torch

from . import rotmat, quat, aaxis

"""
Operations
"""
def interpolate(x_from, x_to, t):
    r_from, p_from = x_from[..., :3, :3], x_from[..., :3, 3]
    r_to,   p_to   = x_to[..., :3, :3], x_to[..., :3, 3]

    r = rotmat.interpolate(r_from, r_to, t)
    p = p_from + (p_to - p_from) * t

    return rotmat.to_xform(r, translation=p)

def fk(local_xforms, root_pos, skeleton):
    """
    Attributes:
        local_xforms: (..., J, 4, 4)
        root_pos: (..., 3), global root position
        skeleton: aPyOpenGL.agl.Skeleton
    """
    pre_xforms = torch.from_numpy(skeleton.pre_xforms).to(local_xforms.device) # (J, 4, 4)
    pre_xforms = torch.tile(pre_xforms, local_xforms.shape[:-3] + (1, 1, 1)) # (..., J, 4, 4)
    pre_xforms[..., 0, :3, 3] = root_pos
    
    global_xforms = [pre_xforms[..., 0, :, :] @ local_xforms[..., 0, :, :]]
    for i in range(1, skeleton.num_joints):
        global_xforms.append(global_xforms[skeleton.parent_idx[i]] @ pre_xforms[..., i, :, :] @ local_xforms[..., i, :, :])
    
    global_xforms = torch.stack(global_xforms, dim=-3) # (..., J, 4, 4)
    return global_xforms

"""
Transformation matrix to other representation
"""
def to_rotmat(xform):
    return xform[..., :3, :3].clone()

def to_quat(xform):
    return rotmat.to_quat(to_rotmat(xform))

def to_aaxis(xform):
    return quat.to_aaxis(to_quat(xform))

def to_ortho6d(xform):
    return rotmat.to_ortho6d(to_rotmat(xform))

def to_translation(xform):
    return xform[..., :3, 3].clone()

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