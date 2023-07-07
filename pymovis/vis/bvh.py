import os
import re
import numpy as np
import multiprocessing as mp

from .motion import Skeleton, Pose, Motion
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

class BVH:
    def __init__(self, filename: str, target_fps=30, to_meter=0.01, v_up=npconst.UP(), v_forward=npconst.FORWARD()):
        self.filename = filename
        self.target_fps = target_fps
        self.to_meter = to_meter
        self.v_up = v_up
        self.v_forward = v_forward

        self.poses = []
        self._load()
    
    def _load(self):
        if not self.filename.endswith(".bvh"):
            print(f"{self.filename} is not a bvh file.")
            return
        
        v_up = np.array(self.v_up)
        v_forward = np.array(self.v_forward)

        assert v_up.shape == (3,) and v_forward.shape == (3,), f"v_up and v_forward must be 3D vectors, but got {v_up.shape} and {v_forward.shape}."

        i = 0
        active = -1
        end_site = False

        skeleton = Skeleton(joints=[], v_up=v_up, v_forward=v_forward)

        with open(self.filename, "r") as f:
            for line in f:
                if "HIERARCHY" in line: continue
                if "MOTION" in line: continue
                if "{" in line: continue

                rmatch = re.match(r"ROOT (\w+)", line)
                if rmatch:
                    skeleton.add_joint(rmatch.group(1), parent_idx=None)
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
                        skeleton.joints[active].local_p = np.array(list(map(float, offmatch.groups())), dtype=np.float32) * self.to_meter
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
                    skeleton.add_joint(jmatch.group(1), parent_idx=active)
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
                    if fps % self.target_fps != 0:
                        raise Exception(f"Invalid target fps for {self.filename}: {self.target_fps} (fps: {fps})")

                    sampling_step = fps // self.target_fps
                    continue

                dmatch = line.strip().split(' ')
                if dmatch:
                    data_block = np.array(list(map(float, dmatch)), dtype=np.float32)
                    N = skeleton.num_joints
                    fi = i
                    if channels == 3:
                        positions[fi, 0:1] = data_block[0:3] * self.to_meter
                        rotations[fi, :]   = data_block[3:].reshape(N, 3)
                    elif channels == 6:
                        data_block         = data_block.reshape(N, 6)
                        positions[fi, :]   = data_block[:, 0:3] * self.to_meter
                        rotations[fi, :]   = data_block[:, 3:6]
                    elif channels == 9: 
                        positions[fi, 0]   = data_block[0:3] * self.to_meter
                        data_block         = data_block[3:].reshape(N - 1, 9)
                        rotations[fi, 1:]  = data_block[:, 3:6]
                        positions[fi, 1:]  += data_block[:, 0:3] * data_block[:, 6:9]
                    else:
                        raise Exception(f"Invalid channels: {channels}")

                    self.poses.append(Pose.from_bvh(skeleton, rotations[fi], order, positions[fi, 0]))
                    i += 1

        self.poses = self.poses[1::sampling_step]
    
    def motion(self):
        name = os.path.splitext(os.path.basename(self.filename))[0]
        res = Motion(self.poses, fps=self.target_fps, name=name)
        return res