import torch
import numpy as np

from pymovis.ops import rotation
from pymovis.utils import npconst

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
    global_R, global_p = R_fk_torch(local_R, root_p, skeleton)
    return rotation.R_to_R6(global_R), global_p

def R6_fk_numpy(local_R6, root_p, skeleton):
    local_R = rotation.R6_to_R(local_R6)
    global_R, global_p = R_fk_numpy(local_R, root_p, skeleton)
    return rotation.R_to_R6(global_R), global_p

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

####################################################################################

def scaled_motion(
    motion,
    scale_factor,
    tolerance=0.1,
    scale_root_by="velocity",
    scale_effector_by="position",
    left_leg_name="LeftUpLeg",
    left_foot_name="LeftFoot",
    right_leg_name="RightUpLeg",
    right_foot_name="RightFoot",
):
    """
    Returns a scaled motion if the scale factor is proper.
    Otherwise, returns None.
    """
    ret_motion = motion.copy()
    if scale_factor == 1.0:
        return ret_motion
    
    # joint indices
    left_leg_idx   = motion.skeleton.idx_by_name[left_leg_name]
    left_foot_idx  = motion.skeleton.idx_by_name[left_foot_name]
    left_knee_idx  = motion.skeleton.parent_idx[left_foot_idx]

    right_leg_idx  = motion.skeleton.idx_by_name[right_leg_name]
    right_foot_idx = motion.skeleton.idx_by_name[right_foot_name]
    right_knee_idx = motion.skeleton.parent_idx[right_foot_idx]

    # chain lengths
    def chain_len(indices):
        return np.sum([np.linalg.norm(motion.skeleton.joints[idx].offset, axis=-1) for idx in indices])
    
    left_lower_chain_len = chain_len([left_knee_idx, left_foot_idx])
    right_lower_chain_len = chain_len([right_knee_idx, right_foot_idx])

    # scale root
    if scale_root_by == "position":
        root_p = np.stack([pose.root_p for pose in ret_motion.poses], axis=0)
        root_p = root_p * np.array([scale_factor, 1.0, scale_factor], dtype=np.float32)
        for i in range(len(ret_motion)):
            ret_motion.poses[i].set_root_p(root_p[i])

    elif scale_root_by == "velocity":
        root_v = np.stack([(ret_motion.poses[i+1].root_p - ret_motion.poses[i].root_p) * scale_factor for i in range(len(ret_motion.poses) - 1)], axis=0)
        root_v = root_v * np.array([1.0, 0.0, 1.0], dtype=np.float32)
        for i in range(1, len(ret_motion)):
            root_p = ret_motion.poses[i-1].root_p + root_v[i-1]
            root_p[1] = ret_motion.poses[i].root_p[1]
            ret_motion.poses[i].set_root_p(root_p)
    
    else:
        raise ValueError(f"scale_root is supposed to be either 'position' or 'velocity', but got {scale_root_by}")
    
    # scale effector
    def too_far(scaled_p, base_idx, chain_len):
        return np.any(np.linalg.norm(scaled_p - copy_global_ps[:, base_idx], axis=-1) > chain_len + tolerance)
    def too_close(left_foot_p, right_foot_p):
        return np.any(np.linalg.norm(left_foot_p - right_foot_p, axis=-1) < tolerance)

    if scale_effector_by == "position":
        global_ps = np.stack([pose.global_p for pose in ret_motion.poses])
        def root_to_joint(joint_idx, scale_y=False):
            res = global_ps[:, joint_idx] - global_ps[:, 0]
            if scale_y:
                return res * scale_factor
            else:
                return res * npconst.Y() + (res * npconst.XZ()) * scale_factor

        copy_global_ps = np.stack([pose.global_p for pose in ret_motion.poses])
        scaled_left_foot  = copy_global_ps[:, 0] + root_to_joint(left_foot_idx)
        scaled_right_foot = copy_global_ps[:, 0] + root_to_joint(right_foot_idx)

        # discard if the target foot position is too far or too close
        if too_far(scaled_left_foot, left_leg_idx, left_lower_chain_len) or too_far(scaled_right_foot, right_leg_idx, right_lower_chain_len):
            print(f"Scale factor {scale_factor} is too large.")
            return None
        if too_close(scaled_left_foot, scaled_right_foot):
            print(f"Scale factor {scale_factor} is too small.")
            return None

        # apply two-bone IK
        for i in range(len(ret_motion)):
            ret_motion.poses[i].two_bone_ik(left_leg_idx,  left_foot_idx,  scaled_left_foot[i])
            ret_motion.poses[i].two_bone_ik(right_leg_idx, right_foot_idx, scaled_right_foot[i])
            ret_motion.poses[i].update()
    
    elif scale_effector_by == "velocity":
        # scale root relative velocity
        rel_ps = np.stack([pose.global_p - pose.root_p for pose in ret_motion.poses])
        rel_vs = (rel_ps[1:] - rel_ps[:-1]) * npconst.Y() + (rel_ps[1:] - rel_ps[:-1]) * npconst.XZ() * scale_factor
        for i in range(1, len(ret_motion)):
            left_foot_p  = rel_vs[i-1, left_foot_idx] + ret_motion.poses[i].root_p + ret_motion.poses[i-1].global_p[left_foot_idx] - ret_motion.poses[i-1].root_p
            right_foot_p = rel_vs[i-1, right_foot_idx] + ret_motion.poses[i].root_p + ret_motion.poses[i-1].global_p[right_foot_idx] - ret_motion.poses[i-1].root_p
            left_foot_p[1] = ret_motion.poses[i].global_p[left_foot_idx][1]
            right_foot_p[1] = ret_motion.poses[i].global_p[right_foot_idx][1]

            # discard if the target foot position is too far
            if too_far(left_foot_p, left_leg_idx, left_lower_chain_len) or too_far(right_foot_p, right_leg_idx, right_lower_chain_len):
                print(f"Scale factor {scale_factor} is too large.")
                return None
            if too_close(left_foot_p, right_foot_p):
                print(f"Scale factor {scale_factor} is too small.")
                return None
            
            # apply two-bone IK
            ret_motion.poses[i].two_bone_ik(left_leg_idx,  left_foot_idx,  left_foot_p)
            ret_motion.poses[i].two_bone_ik(right_leg_idx, right_foot_idx, right_foot_p)
            ret_motion.poses[i].update()
    
    else:
        raise ValueError(f"scale_effector is supposed to be either 'position' or 'velocity', but got {scale_effector_by}")

    return ret_motion

####################################################################################

def get_local_velocity(motion):
    """
    Base-relative velocity
    """
    v = np.stack([motion.poses[i+1].global_p - motion.poses[i].global_p for i in range(len(motion.poses) - 1)])
    v = np.concatenate([v[0:1], v], axis=0) # (T, J, 3)

    # local-to-global rotation matrix
    up = np.stack([motion.poses[i].up for i in range(len(motion.poses))])
    forward = np.stack([motion.poses[i].forward for i in range(len(motion.poses))])
    left = np.stack([motion.poses[i].left for i in range(len(motion.poses))])
    R = np.stack([left, up, forward], axis=-1)  # (T, 3, 3)

    # global-to-local transformation
    v = np.einsum("Trc,Tjc->Tjr", R.transpose((0, 2, 1)), v) # (T, J, 3)
    
    return v

def get_local_position(motion):
    """
    Base-relative position
    """
    p = np.stack([motion.poses[i].global_p - motion.poses[i].base for i in range(len(motion.poses))])
    
    # local-to-global rotation matrix
    up = np.stack([motion.poses[i].up for i in range(len(motion.poses))])
    forward = np.stack([motion.poses[i].forward for i in range(len(motion.poses))])
    left = np.stack([motion.poses[i].left for i in range(len(motion.poses))])
    R = np.stack([left, up, forward], axis=-1)  # (T, 3, 3)

    # global-to-local transformation
    p = np.einsum("Trc,Tjc->Tjr", R.transpose((0, 2, 1)), p) # (T, J, 3)
    
    return p