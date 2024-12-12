import numpy as np

from . import quat, aaxis, ortho6d, xform, euler

"""
Operations
"""
def interpolate(r_from, r_to, t):
    q_from = to_quat(r_from)
    q_to   = to_quat(r_to)
    q = quat.interpolate(q_from, q_to, t)
    return quat.to_rotmat(q)

def fk(local_rotmats, root_pos, skeleton):
    """
    Attributes:
        local_rotmats: (..., J, 3, 3)
        root_pos: (..., 3), global root position
        skeleton: aPyOpenGL.agl.Skeleton
    """
    pre_xforms = np.tile(skeleton.pre_xforms, local_rotmats.shape[:-3] + (1, 1, 1)) # (..., J, 4, 4)
    pre_rotmats = xform.to_rotmat(pre_xforms) # (..., J, 3, 3)
    pre_pos     = xform.to_translation(pre_xforms) # (..., J, 3)
    pre_pos[..., 0, :] = root_pos

    global_rotmats = [np.matmul(pre_rotmats[..., 0, :, :], local_rotmats[..., 0, :, :])]
    global_pos = [pre_pos[..., 0, :]]

    for i in range(1, skeleton.num_joints):
        parent_idx = skeleton.parent_idx[i]
        global_rotmats.append(np.matmul(np.matmul(global_rotmats[parent_idx], pre_rotmats[..., i, :, :]), local_rotmats[..., i, :, :]))
        global_pos.append(np.einsum("...ij,...j->...i", global_rotmats[parent_idx], pre_pos[..., i, :]) + global_pos[parent_idx])
    
    global_rotmats = np.stack(global_rotmats, axis=-3) # (..., J, 3, 3)
    global_pos = np.stack(global_pos, axis=-2) # (..., J, 3)

    return global_rotmats, global_pos

def inv(r):
    return np.transpose(r, axes=(-2, -1))

"""
Rotation matrix to other representation
"""
def to_aaxis(rotmat):
    return quat.to_aaxis(to_quat(rotmat))

def to_quat(rotmat):
    batch_dim = rotmat.shape[:-2]
    rotmat_ = rotmat.reshape(batch_dim + (9,))
    rotmat00, rotmat01, rotmat02, rotmat10, rotmat11, rotmat12, rotmat20, rotmat21, rotmat22 = rotmat_[..., 0], rotmat_[..., 1], rotmat_[..., 2], rotmat_[..., 3], rotmat_[..., 4], rotmat_[..., 5], rotmat_[..., 6], rotmat_[..., 7], rotmat_[..., 8]

    def _to_positive_sqrt(x):
        ret = np.zeros_like(x)
        positive = x > 0
        ret[positive] = np.sqrt(x[positive])
        return ret

    quat_square = np.stack([
        (1.0 + rotmat00 + rotmat11 + rotmat22), # 4*r*r
        (1.0 + rotmat00 - rotmat11 - rotmat22), # 4*i*i
        (1.0 - rotmat00 + rotmat11 - rotmat22), # 4*j*j
        (1.0 - rotmat00 - rotmat11 + rotmat22), # 4*k*k
    ], axis=-1) # (..., 4)
    quat_abs = _to_positive_sqrt(quat_square) # 2*|r|, 2*|i|, 2*|j|, 2*|k|
    r, i, j, k = quat_abs[..., 0], quat_abs[..., 1], quat_abs[..., 2], quat_abs[..., 3]

    quat_candidates = np.stack([
        np.stack([r*r, rotmat21-rotmat12, rotmat02-rotmat20, rotmat10-rotmat01], axis=-1),
        np.stack([rotmat21-rotmat12, i*i, rotmat01+rotmat10, rotmat02+rotmat20], axis=-1),
        np.stack([rotmat02-rotmat20, rotmat01+rotmat10, j*j, rotmat12+rotmat21], axis=-1),
        np.stack([rotmat10-rotmat01, rotmat02+rotmat20, rotmat12+rotmat21, k*k], axis=-1),
    ], axis=-2) # (..., 4, 4)
    quat_candidates = quat_candidates / (2 * quat_abs[..., None] + 1e-8)

    quat_idx = np.argmax(quat_square, axis=-1)
    quat = np.take_along_axis(quat_candidates, quat_idx[..., None, None].repeat(4, axis=-1), axis=-2).squeeze(-2)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)

    return quat.reshape(batch_dim + (4,))

def to_ortho6d(rotmat):
    return np.concatenate([rotmat[..., 0, :], rotmat[..., 1, :]], axis=-1)

def to_xform(rotmat, translation=None):
    batch_dims = rotmat.shape[:-2]

    # transformation matrix
    I = np.eye(4, dtype=np.float32) # (4, 4)
    I = np.tile(I, reps=batch_dims + (1, 1)) # (..., 4, 4)

    # fill rotation matrix
    I[..., :3, :3] = rotmat # (..., 4, 4)

    # fill translation
    if translation is not None:
        I[..., :3, 3] = translation

    return I

def to_euler(rotmat, order, radians=True):
    """
    Assumes extrinsic rotation and Tait-Bryan angles.
    Alpha, beta, gamma are the angles of rotation about the x, y, z axes respectively.
    TODO: handle gimbal lock (singularities)
    """
    if len(order) != 3:
        raise ValueError(f"Order must be a 3-element list, but got {len(order)} elements")
    
    order = order.lower()
    if set(order) != set("xyz"):
        raise ValueError(f"Invalid order: {order}")
    
    axis2idx = {"x": 0, "y": 1, "z": 2}
    idx0, idx1, idx2 = (axis2idx[axis] for axis in order)

    # compute beta
    sign = -1.0 if (idx0 - idx2) % 3 == 2 else 1.0
    beta = np.arcsin(sign * rotmat[..., idx0, idx2])

    # compute alpha
    sign = -1.0 if (idx0 - idx2) % 3 == 1 else 1.0
    alpha = np.arctan2(sign * rotmat[..., idx1, idx2], rotmat[..., idx2, idx2])

    # compute gamma -> same sign as alpha
    gamma = np.arctan2(sign * rotmat[..., idx0, idx1], rotmat[..., idx0, idx0])

    if not radians:
        alpha, beta, gamma = np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma)

    return np.stack([alpha, beta, gamma], axis=-1)

"""
Other representation to rotation matrix
"""
def from_aaxis(a):
    return aaxis.to_rotmat(a)

def from_quat(q):
    return quat.to_rotmat(q)

def from_ortho6d(r):
    return ortho6d.to_rotmat(r)

def from_xform(x):
    return xform.to_rotmat(x)

def from_euler(angles, order, radians=True):
    return euler.to_rotmat(angles, order, radians=radians)