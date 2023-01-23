import torch
import glfw
import glm
import copy

import numpy as np

from pymovis.motion.data import bvh
from pymovis.motion.core import Motion

from pymovis.vis.render import Render
from pymovis.vis.app import App, MotionApp
from pymovis.vis.appmanager import AppManager

class SimpleApp(App):
    def __init__(self):
        super().__init__()
        self.m = Render.plane()
    
    def render(self):
        self.m.draw()

class MyApp(MotionApp):
    def __init__(self, motion: Motion, vel_factor):
        super().__init__(motion)
        self.motion = motion
        self.vel_factor = vel_factor

        self.left_leg_idx   = self.motion.skeleton.idx_by_name["LeftUpLeg"]
        self.left_foot_idx  = self.motion.skeleton.idx_by_name["LeftFoot"]
        self.right_leg_idx  = self.motion.skeleton.idx_by_name["RightUpLeg"]
        self.right_foot_idx = self.motion.skeleton.idx_by_name["RightFoot"]

        # heightmap
        # self.heightmap = Heightmap.load_from_file("./data/heightmaps/hmap_010_smooth.txt")
        # self.heightmap_mesh = Render.mesh(self.heightmap.mesh).set_texture("grid.png").set_uv_repeat(0.1)

        # grid for environment map
        grid_x = np.linspace(-1, 1, 11)
        grid_z = np.linspace(-1, 1, 11)
        grid_x, grid_z = np.meshgrid(grid_x, grid_z)
        grid_y = np.zeros_like(grid_x)
        self.env_map = np.stack([grid_x, grid_y, grid_z], axis=-1)
        self.sphere = Render.sphere(0.05).set_color([0, 1, 0])
        # self.cubemap = Render.cubemap("skybox")

        # velocity-based locomotion scaling
        self.scale_motion()
    
    # TODO: implement this
    def scale_motion(self):
        self.dupl_motions = []
        for vf in self.vel_factor:
            dupl_motion = copy.deepcopy(self.motion)
            root_v = (self.motion.root_p[1:] - self.motion.root_p[:-1]) * vf * np.array([1, 0, 1])
            for i in range(1, len(dupl_motion.root_p)):
                dupl_motion.root_p[i] = dupl_motion.root_p[i-1] + root_v[i-1]
                dupl_motion.root_p[i, 1] = self.motion.root_p[i, 1]
            dupl_motion.update()

            base_to_left_foot  = self.motion.global_p[:, self.left_foot_idx] - self.motion.root_p
            base_to_right_foot = self.motion.global_p[:, self.right_foot_idx] - self.motion.root_p

            base_to_left_foot = base_to_left_foot * np.array([0, 1, 0]) + (base_to_left_foot * np.array([1, 0, 1])) * vf
            base_to_right_foot = base_to_right_foot * np.array([0, 1, 0]) + (base_to_right_foot * np.array([1, 0, 1])) * vf

            dupl_motion.two_bone_ik(self.left_leg_idx, self.left_foot_idx, base_to_left_foot + dupl_motion.root_p)
            dupl_motion.two_bone_ik(self.right_leg_idx, self.right_foot_idx, base_to_right_foot + dupl_motion.root_p)

            self.dupl_motions.append(dupl_motion)


    def render(self):
        super().render()
        # self.heightmap_mesh.draw()
        # r = np.stack([self.motion.poses[self.frame].left, self.motion.poses[self.frame].up, self.motion.poses[self.frame].forward], axis=-1)
        # env_map = np.einsum("ij,abj->abi", r, self.env_map) + self.motion.poses[self.frame].base
        # env_map = np.reshape(env_map, [-1, 3])
        # env_map[..., 1] = self.heightmap.sample_height(env_map[..., 0], env_map[..., 2])
        # for e in env_map:
        #     self.sphere.set_position(e).draw()
        
        for i, m in enumerate(self.dupl_motions):
            m.render_by_frame(self.frame, (i + 1) * 0.25)
        # self.cubemap.draw()


if __name__ == "__main__":
    motion = bvh.load("./data/animations/PFNN_LocomotionFlat01_000.bvh", v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01)
    # motion = bvh.load("D:/data/LaFAN1/aiming1_subject1.bvh", v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01)
    motion.align_by_frame(0)

    app_manager = AppManager()
    app = MyApp(motion, [0.8, 0.9, 1.1, 1.2])
    app_manager.run(app)

    app_manager = AppManager()
    app = SimpleApp()
    app_manager.run(app)