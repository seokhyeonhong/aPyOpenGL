import os
import glfw
import glm

from pymovis import agl

class MotionApp(agl.AnimApp):
    def __init__(self, bvh_filename):
        super().__init__()

        # motion data
        bvh = agl.BVH(bvh_filename)
        self.motion = bvh.motion()
        self.model = bvh.model()
        self.total_frames = self.motion.num_frames()
        self.fps = self.motion.fps

        # camera options
        self.focus_on_root = False
        self.follow_root = False
        self.init_cam_pos = self.camera.position
    
    def start(self):
        super().start()

        # render options
        self.render_skeleton = agl.Render.skeleton(self.model)

        # UI options
        self.ui.add_menu("MotionApp")
        self.ui.add_menu_item("MotionApp", "X-Ray", self.render_skeleton.switch_visible, key=glfw.KEY_X)

    def update(self):
        super().update()

        # set camera focus on the root
        if self.focus_on_root:
            self.camera.set_focus_position(self.motion.poses[self.curr_frame].root_p)
            self.camera.set_up(glm.vec3(0, 1, 0))
        elif self.follow_root:
            self.camera.set_position(self.motion.poses[self.curr_frame].root_p + glm.vec3(0, 1.5, 5))
            self.camera.set_focus_position(self.motion.poses[self.curr_frame].root_p)
            self.camera.set_up(glm.vec3(0, 1, 0))
        self.camera.update()
        
        # set pose
        self.model.set_pose(self.motion.poses[self.curr_frame])

    def render(self):
        super().render()

        # render the environment
        self.grid.draw()
        self.axis.draw()

    def render_xray(self):
        super().render_xray()
        self.render_skeleton.update_skeleton(self.model).draw()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        # set camera focus on the root
        if key == glfw.KEY_F3 and action == glfw.PRESS:
            self.focus_on_root = not self.focus_on_root
        elif key == glfw.KEY_F4 and action == glfw.PRESS:
            self.follow_root = not self.follow_root
            
if __name__ == "__main__":
    bvh_filename = os.path.join(agl.AGL_PATH, "data/bvh/ybot_capoeira.bvh")
    agl.AppManager.start(MotionApp(bvh_filename))