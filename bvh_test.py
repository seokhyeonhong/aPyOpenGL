import torch
import glfw
import glm
import copy

import numpy as np

from pymovis.motion.data import bvh
from pymovis.motion.core import Motion
from pymovis.motion.ops import npmotion

from pymovis.vis.core import Vertex, VAO, Mesh
from pymovis.vis.render import Render
from pymovis.vis.app import MotionApp
from pymovis.vis.appmanager import AppManager
from pymovis.vis.heightmap import Heightmap

class MyApp(MotionApp):
    def __init__(self, motion: Motion, vel_factor):
        super().__init__(motion)
        self.motion = motion
        self.vel_factor = vel_factor

        self.left_leg_idx = self.motion.skeleton.idx_by_name["LeftUpLeg"]
        self.left_foot_idx = self.motion.skeleton.idx_by_name["LeftFoot"]
        self.right_leg_idx = self.motion.skeleton.idx_by_name["RightUpLeg"]
        self.right_foot_idx = self.motion.skeleton.idx_by_name["RightFoot"]

        self.heightmap = Heightmap.load_from_file("./data/heightmaps/hmap_010_smooth.txt")
        self.heightmap_mesh = Render.mesh(self.heightmap.mesh).set_texture("grid.png").set_uv_repeat(0.1)

        grid_x = np.linspace(-1, 1, 11)
        grid_z = np.linspace(-1, 1, 11)
        grid_x, grid_z = np.meshgrid(grid_x, grid_z)
        grid_y = np.zeros_like(grid_x)
        self.env_map = np.stack([grid_x, grid_y, grid_z], axis=-1)
        self.sphere = Render.sphere(0.05).set_material([0, 1, 0])
    
    def render(self):
        super().render()
        self.heightmap_mesh.draw()
        # print()
        r = np.stack([self.motion.poses[self.frame].left, self.motion.poses[self.frame].up, self.motion.poses[self.frame].forward], axis=-1)
        env_map = np.einsum("ij,abj->abi", r, self.env_map) + self.motion.poses[self.frame].base
        env_map = np.reshape(env_map, [-1, 3])
        env_map[..., 1] = self.heightmap.sample_height(env_map[..., 0], env_map[..., 2])
        for e in env_map:
            self.sphere.set_position(e).draw()


if __name__ == "__main__":
    motion = bvh.load("D:/data/LaFAN1/aiming1_subject1.bvh", v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01)
    motion.align_by_frame(0)

    app_manager = AppManager()
    app = MyApp(motion, [1.2])
    app_manager.run(app)