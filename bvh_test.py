import torch
import glfw
import glm
import copy

import numpy as np

from pymovis.motion.data import bvh
from pymovis.motion.core import Motion
from pymovis.vis.render import Render
from pymovis.vis.app import App
from pymovis.vis.appmanager import AppManager

class MotionApp(App):
    def __init__(self, motion: Motion, motion2: Motion):
        super().__init__()
        self.motion = motion
        self.motion2 = motion2
        self.pose = self.motion.poses[0]

    def render(self):
        Render.plane().set_texture("example.png").set_scale(50).set_uv_repeat(5).draw()
        Render.arrow().draw()
        Render.text(int(glfw.get_time() * self.motion.fps)).draw()
        
        self.motion.render_by_time(glfw.get_time())
        self.motion2.render_by_time(glfw.get_time())
        # self._camera.focus_position = glm.vec3(self.motion.get_pose_by_time(glfw.get_time()).root_p.numpy())

if __name__ == "__main__":
    # import time
    # t = time.time()
    motion = bvh.load("D:/data/NSM/Sit/Tall1.bvh", v_forward=[0, -1, 0], v_up=[1, 0, 0])
    motion.align_by_frame(0)

    motion2 = copy.deepcopy(motion)
    motion2.translate_root(motion2.root_p * np.array([1, 0, 1]) * 1.5)

    app_manager = AppManager()
    app = MotionApp(motion, motion2)
    app_manager.run(app)