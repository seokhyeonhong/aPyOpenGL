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

# def clamp(num, min_value, max_value):
#    return max(min(num, max_value), min_value)

# class MotionApp(App):
#     def __init__(self, motion: Motion, vel_factor):
#         super().__init__()
#         self.motion = motion

#         _, global_p = npmotion.R.fk(self.motion.local_R, self.motion.root_p, self.motion.skeleton)
#         effector_p = global_p#[:, self.motion.skeleton.effector_idx]
#         self.root_ps, self.effector_ps = self.__init_positions(effector_p, vel_factor)

#         self.sphere = Render.sphere(radius=0.05).set_material(albedo=glm.vec3(.5))
    
#     def __init_positions(self, effector_p, vel_factor):
#         root_ps = [self.motion.root_p[0]]
#         effector_ps = [effector_p[0]]
#         for i in range(1, len(self.motion)):
#             root_v = self.motion.root_p[i] - self.motion.root_p[i-1]
#             effector_v = effector_p[i] - effector_p[i-1] - root_v
#             root_ps.append(vel_factor * root_v + root_ps[-1])
#             effector_ps.append(vel_factor * (effector_v + root_v) + effector_ps[-1])
#         return np.array(root_ps), np.array(effector_ps)

#     def render(self):
#         Render.plane().set_texture("example.png").set_scale(50).set_uv_repeat(5).draw()
#         Render.arrow().draw()

#         frame = clamp(int(glfw.get_time() * self.motion.fps), 0, len(self.motion) - 1)
#         # self.camera.set_focus_position(self.root_ps[frame])
#         Render.text(frame).draw()
#         self.sphere.set_position(self.root_ps[frame]).draw()
#         for effector in self.effector_ps[frame]:
#             self.sphere.set_position(effector).draw()
#         self.motion.render_by_frame(frame)

if __name__ == "__main__":
    # import time
    # t = time.time()
    motion = bvh.load("D:/data/LaFAN1/walk1_subject1.bvh", v_forward=[0, 1, 0], v_up=[1, 0, 0])
    motion.align_by_frame(0)

    app_manager = AppManager()
    app = MotionApp(motion)
    app_manager.run(app)