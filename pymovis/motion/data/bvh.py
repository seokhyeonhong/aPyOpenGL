import os
import re

import numpy as np
import multiprocessing as mp

from pymovis.motion.core import Skeleton, Pose, Motion
from pymovis.utils import npconst, util

channelmap = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2,
}

def load(filename, target_fps=30, to_meter=0.01, v_up=npconst.UP(), v_forward=npconst.FORWARD()):
    if not filename.endswith(".bvh"):
        print(f"{filename} is not a bvh file.")
        return
    
    v_up = np.array(v_up)
    v_forward = np.array(v_forward)

    assert v_up.shape == (3,) and v_forward.shape == (3,), f"v_up and v_forward must be 3D vectors, but got {v_up.shape} and {v_forward.shape}."

    i = 0
    active = -1
    end_site = False

    skeleton = Skeleton(joints=[], v_up=v_up, v_forward=v_forward)
    poses = []

    with open(filename, "r") as f:
        for line in f:
            if "HIERARCHY" in line: continue
            if "MOTION" in line: continue
            if "{" in line: continue

            rmatch = re.match(r"ROOT (\w+)", line)
            if rmatch:
                skeleton.add_joint(rmatch.group(1), None)
                active = skeleton.num_joints - 1
                continue

            if "}" in line:
                if end_site:
                    end_site = False
                else:
                    active = skeleton.parent_idx[active]
                continue

            offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
            if offmatch:
                if not end_site:
                    skeleton.joints[active].offset = np.array(list(map(float, offmatch.groups())), dtype=np.float32) * to_meter
                continue

            chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
            if chanmatch:
                channels = int(chanmatch.group(1))
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
                continue

            jmatch = re.match("\s*JOINT\s+(\w+)", line)
            if jmatch:
                skeleton.add_joint(jmatch.group(1), active)
                active = skeleton.num_joints - 1
                continue

            if "End Site" in line:
                end_site = True
                continue

            fmatch = re.match("\s*Frames:\s+(\d+)", line)
            if fmatch:
                fnum = int(fmatch.group(1))
                positions = np.zeros((fnum, skeleton.num_joints, 3), dtype=np.float32)
                rotations = np.zeros((fnum, skeleton.num_joints, 3), dtype=np.float32)
                continue

            fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
            if fmatch:
                frametime = float(fmatch.group(1))
                fps = round(1. / frametime)
                if fps % target_fps != 0:
                    raise Exception(f"Invalid target fps for {filename}: {target_fps} (fps: {fps})")

                sampling_step = fps // target_fps
                continue

            dmatch = line.strip().split(' ')
            if dmatch:
                data_block = np.array(list(map(float, dmatch)), dtype=np.float32)
                N = skeleton.num_joints
                fi = i
                if channels == 3:
                    positions[fi, 0:1] = data_block[0:3] * to_meter
                    rotations[fi, :]   = data_block[3:].reshape(N, 3)
                elif channels == 6:
                    data_block         = data_block.reshape(N, 6)
                    positions[fi, :]   = data_block[:, 0:3] * to_meter
                    rotations[fi, :]   = data_block[:, 3:6]
                elif channels == 9: 
                    positions[fi, 0]   = data_block[0:3] * to_meter
                    data_block         = data_block[3:].reshape(N - 1, 9)
                    rotations[fi, 1:]  = data_block[:, 3:6]
                    positions[fi, 1:]  += data_block[:, 0:3] * data_block[:, 6:9]
                else:
                    raise Exception(f"Invalid channels: {channels}")

                poses.append(Pose.from_bvh(skeleton, rotations[fi], order, positions[fi, 0]))
                i += 1

    poses = poses[1::sampling_step]
    name = os.path.splitext(os.path.basename(filename))[0]
    return Motion(name=name, skeleton=skeleton, poses=poses, global_v=None, fps=target_fps)

def load_parallel(files, cpus=mp.cpu_count(), **kwargs):
    return util.run_parallel(load, files, cpus, desc=f"Loading {len(files)} BVH files", **kwargs)