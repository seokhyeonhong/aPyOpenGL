import torch
import glfw
import glm

from pymovis.motion.data import bvh
from pymovis.motion.core.motion import Motion
from pymovis.vis.render import Render
from pymovis.vis.app import App
from pymovis.vis.appmanager import AppManager

class MotionApp(App):
    def __init__(self, motion: Motion):
        super().__init__()
        self.__a = 5
        self.motion = motion
        self.pose = self.motion.poses[0]

    def render(self):
        Render.plane().set_texture("example.png").set_scale(50).set_uv_repeat(5).draw()
        Render.arrow().draw()
        
        self.motion.render_by_time(glfw.get_time())
        # self._camera.focus_position = glm.vec3(self.motion.get_pose_by_time(glfw.get_time()).root_p.numpy())

if __name__ == "__main__":
    # import time
    # t = time.time()
    motion = bvh.load("pymovis/motion/data/sample.bvh")
    motion.align_by_frame(100)
    print(motion.poses[100].forward)
    # print(time.time() - t)
    # exit()
    
    # # check how many memory is used

    app = MotionApp(motion)
    AppManager.run(app)