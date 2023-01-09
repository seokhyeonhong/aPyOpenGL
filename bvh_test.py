import torch
import glfw
import glm
import copy

import numpy as np

from pymovis.motion.data import bvh
from pymovis.motion.core import Motion
from pymovis.motion.ops import npmotion

from pymovis.vis.render import Render
from pymovis.vis.app import MotionApp
from pymovis.vis.appmanager import AppManager

class MyApp(MotionApp):
    def __init__(self, motion: Motion, vel_factor):
        super().__init__(motion)
        self.motion = motion
        self.vel_factor = vel_factor

        self.left_leg_idx = self.motion.skeleton.idx_by_name["LeftUpLeg"]
        self.left_foot_idx = self.motion.skeleton.idx_by_name["LeftFoot"]
        self.right_leg_idx = self.motion.skeleton.idx_by_name["RightUpLeg"]
        self.right_foot_idx = self.motion.skeleton.idx_by_name["RightFoot"]

        # self.scaled_motion = self.get_scaled_motion()
    
    def get_scaled_motion(self):
        # TODO: parallelize this using numpy array functions
        
        motions = []
        for v in self.vel_factor:
            poses = []
            for i in range(len(self.motion)):
                pose = copy.deepcopy(self.motion.poses[i])

                _, global_p = npmotion.R.fk(pose.local_R, pose.root_p, pose.skeleton)
                pose.root_p *= np.array([v, 1, v])
                
                left_foot_p = global_p[self.left_foot_idx] * np.array([v, 1, v])
                pose.two_bone_ik(self.left_leg_idx, self.left_foot_idx, left_foot_p)

                right_foot_p = global_p[self.right_foot_idx] * np.array([v, 1, v])
                pose.two_bone_ik(self.right_leg_idx, self.right_foot_idx, right_foot_p)
                
                poses.append(pose)
            motions.append(Motion(f"scaled_by_{v}", self.motion.skeleton, poses, fps=self.motion.fps))
        return motions
        # _, global_p = npmotion.R.fk(self.motion.local_R, self.motion.root_p, self.motion.skeleton)
        
        # gt_root_p = global_p[:, self.hip_idx]
        # gt_left_foot = global_p[:, self.left_foot_idx]
        # gt_right_foot = global_p[:, self.right_foot_idx]

        # root_p = [global_p[0, self.hip_idx]]
        # left_foot_p = [global_p[0, self.left_foot_idx]]
        # right_foot_p = [global_p[0, self.right_foot_idx]]
        
        # for i in range(1, len(self.motion)):
        #     # root_v = gt_root_p[i] - gt_root_p[i-1]
        #     # root_p.append(root_p[-1] + self.vel_factor * root_v)

        #     # curr_root2foot = npmotion.normalize(gt_left_foot[i] - gt_root_p[i])
        #     # prev_root2foot = npmotion.normalize(gt_left_foot[i-1] - gt_root_p[i-1])
        #     # angle = np.arccos(np.dot(prev_root2foot, curr_root2foot))[..., np.newaxis]
        #     # axis = np.cross(prev_root2foot, curr_root2foot)
        #     # R = npmotion.R.from_A(angle * self.vel_factor, axis)

        #     # left_foot_pos = root_p[-1] + R @ (gt_left_foot[i-1] - gt_root_p[i-1])
        #     # left_foot_pos[1] = gt_left_foot[i, 1]
        #     # left_foot_p.append(left_foot_pos)
        #     root_p.append(gt_root_p[i] * np.array((self.vel_factor, 1, self.vel_factor)))
        #     left_foot_p.append(gt_left_foot[i] * np.array((self.vel_factor, 1, self.vel_factor)))
        #     right_foot_p.append(gt_right_foot[i] * np.array((self.vel_factor, 1, self.vel_factor)))

        # return np.array(root_p, dtype=np.float32), np.array(left_foot_p, dtype=np.float32), np.array(right_foot_p, dtype=np.float32)

    # def render(self):
    #     super().render()
        # for m in self.scaled_motion:
        #     m.render_by_frame(self.frame, 1.0)

if __name__ == "__main__":
    motion = bvh.load("D:/data/LaFAN1/walk1_subject1.bvh", v_forward=[0, 1, 0], v_up=[1, 0, 0])
    motion.align_by_frame(0)

    app_manager = AppManager()
    app = MyApp(motion, [1.2])
    app_manager.run(app)