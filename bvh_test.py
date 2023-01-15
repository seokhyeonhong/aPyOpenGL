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

        # heightmap = Heightmap.load_from_file("./data/heightmaps/hmap_010_smooth.txt")
        # self.heightmap = Render.mesh(heightmap.mesh).set_texture("grid.png").set_uv_repeat(0.1)
    
    def render(self):
        super().render()
        # self.heightmap.draw()

if __name__ == "__main__":
    motion = bvh.load("D:/data/LaFAN1/aiming1_subject1.bvh", v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01)
    motion.align_by_frame(1000)

    app_manager = AppManager(960, 540)
    app = MyApp(motion, [1.2])
    app_manager.run(app)