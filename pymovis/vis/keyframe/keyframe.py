from __future__ import annotations
from enum import Enum

import glm
import numpy as np

from pymovis.ops import rotation
from pymovis.utils import npconst

class KeyInterpType(Enum):
    eUNKNOWN = 0
    eCONSTANT = 1
    eLINEAR = 2
    eCUBIC = 3

class Keyframe:
    def __init__(self, value=0.0, frame=0.0, type=KeyInterpType.eUNKNOWN):
        self.value = value
        self.frame = frame
        self.type  = type

class NodeKeyframes:
    def __init__(self, name, euler_order=glm.ivec3(0), euler=None, pos=None, scale=None):
        self.name        = name
        self.euler_order = euler_order
        self.euler       = [[], [], []] if euler is None else euler # in degrees
        self.pos         = [[], [], []] if pos is None else pos
        self.scale       = [[], [], []] if scale is None else scale

class SceneKeyframes:
    def __init__(self, name, node_keyframes=None, start_frame=0, end_frame=0, fps=30.0):
        self.name           = name
        self.node_keyframes = [] if node_keyframes is None else node_keyframes
        self.start_frame    = start_frame
        self.end_frame      = end_frame
        self.fps            = fps

def search_key(frames: list[Keyframe], target_frame, iter_idx, upper_checked):
    """
    Returns the index of the maximum frame that is less than or equal to the target frame.
    """
    if iter_idx >= len(frames):
        return len(frames) - 1
    if iter_idx < 0:
        return 0
    
    iter_frame = frames[iter_idx].frame
    if iter_frame == target_frame:
        return iter_idx
    elif iter_frame > target_frame:
        return search_key(frames, target_frame, iter_idx - 1, True)
    else:
        if upper_checked:
            return iter_idx
        else:
            return search_key(frames, target_frame, iter_idx + 1, False)

def interpolate_linear(val0, val1, frame, frame0, frame1):
    if frame0 == frame1:
        return val0
    elif frame > frame1:
        return val1
    elif frame < frame0:
        return val0
    
    w0 = (frame1 - frame) / (frame1 - frame0)
    return w0 * val0 + (1.0 - w0) * val1

def _resample_by_keyframes(keyframes: list[Keyframe], frame_idx: list[int]):
    if len(keyframes) == 0:
        return []
    
    new_keys = []
    lower_idx, max_idx = 0, len(keyframes) - 1

    for i in range(len(frame_idx)):
        fi = frame_idx[i]
        lower_idx = search_key(keyframes, fi, lower_idx, False)
        upper_idx = min(lower_idx + 1, max_idx)

        # TODO: support other interpolation types
        # if keyframes[lower_idx].type == KeyInterpType.eLINEAR:
        interp_val = interpolate_linear(keyframes[lower_idx].value, keyframes[upper_idx].value, fi, keyframes[lower_idx].frame, keyframes[upper_idx].frame)
        # else:
        #     raise ValueError(f"Unsupported interpolation type {keyframes[lower_idx].type}. Interpolation type must be linear, and other types are not supported yet.")
        new_kf = Keyframe(value=interp_val, frame=fi, type=keyframes[lower_idx].type)
        new_keys.append(new_kf)

    return new_keys

def _resample_by_node_keyframes(original: NodeKeyframes, frame_idx: list[int]):
    resampled = NodeKeyframes(original.name, euler_order=original.euler_order)
    for i in range(3):
        resampled.euler[i] = _resample_by_keyframes(original.euler[i], frame_idx)
        resampled.pos[i]   = _resample_by_keyframes(original.pos[i], frame_idx)
        resampled.scale[i] = _resample_by_keyframes(original.scale[i], frame_idx)

    return resampled

def _resample_by_scene_keyframes(scene: SceneKeyframes, frame_idx: list[int]):
    resampled = SceneKeyframes(name=scene.name, start_frame=scene.start_frame, end_frame=scene.end_frame, fps=scene.fps)
    for i in range(len(scene.node_keyframes)):
        keys = _resample_by_node_keyframes(scene.node_keyframes[i], frame_idx)
        resampled.node_keyframes.append(keys)
    
    return resampled

def resample(value, frame_idx) -> SceneKeyframes:
    if isinstance(value, SceneKeyframes):
        return _resample_by_scene_keyframes(value, frame_idx)
    elif isinstance(value, NodeKeyframes):
        return _resample_by_node_keyframes(value, frame_idx)
    elif isinstance(value, list):
        return _resample_by_keyframes(value, frame_idx)
    else:
        raise ValueError(f"Unsupported type {type(value)}")

def get_values(keys: list[Keyframe], nof, scale=1.0):
    values = np.zeros(nof, dtype=np.float32)

    if len(keys) == 0:
        return values
    
    if len(keys) != nof:
        raise ValueError(f"Number of keyframes ({len(keys)}) must be equal to the number of frames ({nof}).")

    for i in range(len(keys)):
        values[i] = keys[i].value * scale

    return values

def get_rotations_from_resampled(names: list[str], scene: SceneKeyframes, nof):
    resampled_rotations = []

    # name to node index
    name_to_idx = {}
    for i in range(len(scene.node_keyframes)):
        name_to_idx[scene.node_keyframes[i].name] = i
    
    # iterate
    for i in range(len(names)):
        idx = name_to_idx.get(names[i], None)
        if idx is None:
            rotations = np.stack([npconst.Q_IDENTITY()] * nof, axis=0)
            print(f"Warning: node {names[i]} not found in the scene.")
        else:
            node = scene.node_keyframes[idx]
            order = node.euler_order

            to_rad = np.pi / 180.0
            e0 = get_values(node.euler[order.x], nof, scale=to_rad)
            e1 = get_values(node.euler[order.y], nof, scale=to_rad)
            e2 = get_values(node.euler[order.z], nof, scale=to_rad)
            E = np.stack([e2, e1, e0], axis=1)

            # E2 * E1 * E0
            xyz = "xyz"
            order = xyz[order.z] + xyz[order.y] + xyz[order.x]
            rotations = rotation.E_to_Q(E, order, radians=True)
        
        resampled_rotations.append(rotations)

    resampled_rotations = np.stack(resampled_rotations, axis=1)
    return resampled_rotations

def get_positions_from_resampled(name: str, scene: SceneKeyframes, nof):
    idx = -1
    for i in range(len(scene.node_keyframes)):
        if scene.node_keyframes[i].name == name:
            idx = i
            break
    
    positions = []
    if idx == -1:
        print(f"Warning: node {name} not found in the scene.")
        return positions
    
    node = scene.node_keyframes[idx]

    xs = get_values(node.pos[0], nof)
    ys = get_values(node.pos[1], nof)
    zs = get_values(node.pos[2], nof)

    nof = scene.end_frame - scene.start_frame + 1

    if not (len(xs) == 0 or len(xs) == nof):
        raise ValueError(f"Number of keyframes ({len(xs)}) must be equal to the number of frames ({nof}).")
    if not (len(ys) == 0 or len(ys) == nof):
        raise ValueError(f"Number of keyframes ({len(ys)}) must be equal to the number of frames ({nof}).")
    if not (len(zs) == 0 or len(zs) == nof):
        raise ValueError(f"Number of keyframes ({len(zs)}) must be equal to the number of frames ({nof}).")
    
    for i in range(nof):
        px = xs[i] if len(xs) > 0 else 0.0
        py = ys[i] if len(ys) > 0 else 0.0
        pz = zs[i] if len(zs) > 0 else 0.0
        positions.append(np.array([px, py, pz], dtype=np.float32))

    positions = np.stack(positions, axis=0)
    return positions