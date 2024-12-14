import os
import re
import numpy as np
import multiprocessing as mp

from .motion import Joint, Skeleton, Pose, Motion
from .model  import Model

from aPyOpenGL.transforms import n_euler

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
    """
    !!!
    Disclaimer:
        This implementation is only for character poses with 3D root positions and 3D joint rotations.
        Therefore, joint positions and scales within the BVH file are not considered.
    !!!
    """
    def __init__(self, filename: str, target_fps=30, scale=0.01):
        self.filename = filename
        self.target_fps = target_fps
        self.scale = scale

        self._cumsum_channels = 0
        self._valid_channel_idx = []

        self.poses = []
        self._load()
    
    def _load(self):
        if not self.filename.endswith(".bvh"):
            print(f"{self.filename} is not a bvh file.")
            return
        
        i = 0
        active = -1
        end_site = False

        skeleton = Skeleton(joints=[])

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
                        skeleton.joints[active].local_pos = np.array(list(map(float, offmatch.groups())), dtype=np.float32) * self.scale
                        skeleton.recompute_pre_xform()
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

                    if active == 0:
                        assert channels == 6, f"Root joint must have 6 channels, but got {channels}"
                        self._valid_channel_idx += [i for i in range(channels)]
                    else:
                        self._valid_channel_idx += [i + self._cumsum_channels for i in range(channelis, channelie)]
                    self._cumsum_channels += channels
                    continue

                jmatch = re.match(r"\s*JOINT\s+(.+)", line)
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
                    # positions = np.zeros((fnum, skeleton.num_joints, 3), dtype=np.float32)
                    # rotations = np.zeros((fnum, skeleton.num_joints, 3), dtype=np.float32)
                    continue

                fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
                if fmatch:
                    frametime = float(fmatch.group(1))
                    fps = round(1. / frametime)
                    # if fps % self.target_fps != 0:
                    #     raise Exception(f"Invalid target fps for {self.filename}: {self.target_fps} (fps: {fps})")

                    # sampling_step = fps // self.target_fps
                    continue

                dmatch = line.strip().split(' ')
                if dmatch:
                    data_block = np.array(list(map(float, dmatch)), dtype=np.float32)
                    data_block = data_block[self._valid_channel_idx]

                    root_pos = data_block[0:3] * self.scale
                    joint_rots = data_block[3:].reshape(skeleton.num_joints, 3)
                    local_quats = n_euler.to_quat(joint_rots, order, radians=False)
                    self.poses.append(Pose(skeleton, local_quats, root_pos))
                    i += 1

        # self.poses = self.poses[1::sampling_step]
    
    def motion(self):
        name = os.path.splitext(os.path.basename(self.filename))[0]
        res = Motion(self.poses, fps=self.target_fps, name=name)
        return res
    
    def model(self):
        return Model(meshes=None, skeleton=self.poses[0].skeleton)