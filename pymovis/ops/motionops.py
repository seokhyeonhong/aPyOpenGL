import torch
import numpy as np

from pymovis.ops import rotation

def R_fk_torch(local_R, root_p, skeleton):
    bone_offsets = torch.from_numpy(skeleton.get_bone_offsets()).to(local_R.device)
    parents = skeleton.parent_idx

    global_R, global_p = [local_R[..., 0, :, :]], [root_p]
    for i in range(1, len(parents)):
        global_R.append(torch.matmul(global_R[parents[i]], local_R[..., i, :, :]))
        global_p.append(torch.matmul(global_R[parents[i]], bone_offsets[i]) + global_p[parents[i]])
    
    global_R = torch.stack(global_R, dim=-3) # (..., N, 3, 3)
    global_p = torch.stack(global_p, dim=-2) # (..., N, 3)
    return global_R, global_p

def R_fk_numpy(local_R, root_p, skeleton):
    bone_offsets, parents = skeleton.get_bone_offsets(), skeleton.parent_idx

    global_R, global_p = [local_R[..., 0, :, :]], [root_p]
    for i in range(1, len(parents)):
        global_R.append(np.matmul(global_R[parents[i]], local_R[..., i, :, :]))
        global_p.append(np.matmul(global_R[parents[i]], bone_offsets[i]) + global_p[parents[i]])
    
    global_R = np.stack(global_R, axis=-3) # (..., N, 3, 3)
    global_p = np.stack(global_p, axis=-2) # (..., N, 3)
    return global_R, global_p

def R_fk(local_R, root_p, skeleton):
    """
    Args:
        local_R: (..., J, 3, 3)
        root_p: (..., 3)
        bone_offset: (J, 3)
        parents: (J,)
    Returns:
        Global rotation matrix and position of each joint.
    """

    if isinstance(local_R, np.ndarray):
        return R_fk_numpy(local_R, root_p, skeleton)
    elif isinstance(local_R, torch.Tensor):
        return R_fk_torch(local_R, root_p, skeleton)
    else:
        raise ValueError(f"Invalid type {type(local_R)}")

####################################################################################

def R6_fk_torch(local_R6, root_p, skeleton):
    local_R = rotation.R6_to_R(local_R6)
    return R_fk_torch(local_R, root_p, skeleton)

def R6_fk_numpy(local_R6, root_p, skeleton):
    local_R = rotation.R6_to_R(local_R6)
    return R_fk_numpy(local_R, root_p, skeleton)

def R6_fk(local_R6, root_p, skeleton):
    """
    Args:
        local_R6: (..., J, 6)
        root_p: (..., 3)
        bone_offset: (J, 3)
        parents: (J,)
    Returns:
        Global rotation matrix and position of each joint.
    """
    if isinstance(local_R6, np.ndarray):
        return R6_fk_numpy(local_R6, root_p, skeleton)
    elif isinstance(local_R6, torch.Tensor):
        return R6_fk_torch(local_R6, root_p, skeleton)
    else:
        raise ValueError(f"Invalid type {type(local_R6)}")