import torch
import glfw
import glm

from pymovis.motion.data import bvh
from pymovis.motion.core import Motion
from pymovis.vis.render import Render
from pymovis.vis.app import App
from pymovis.vis.appmanager import AppManager

class MotionApp(App):
    def __init__(self, motion: Motion):
        super().__init__()
        self.motion = motion
        self.pose = self.motion.poses[0]

    def render(self):
        Render.plane().set_texture("example.png").set_scale(50).set_uv_repeat(5).draw()
        Render.arrow().draw()
        Render.text(int(glfw.get_time() * 30)).draw()
        
        self.motion.render_by_time(glfw.get_time())
        # self._camera.focus_position = glm.vec3(self.motion.get_pose_by_time(glfw.get_time()).root_p.numpy())

if __name__ == "__main__":
    # import time
    # t = time.time()
    motion = bvh.load("D:/data/NSM/Avoid/Avoid1.bvh", v_forward=[0, -1, 0], v_up=[1, 0, 0])
    motion.align_by_frame(0)
    print(motion.poses[0].forward)

    app_manager = AppManager.initialize()
    app = MotionApp(motion)
    app_manager.run(app)