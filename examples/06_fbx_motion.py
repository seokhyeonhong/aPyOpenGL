import os
import glfw
import glm

from aPyOpenGL import agl

class MotionApp(agl.App):
    def __init__(self, motion_filename, model_filename):
        super().__init__()

        # motion data
        self.motion = agl.FBX(motion_filename).motions()[0]
        self.model  = agl.FBX(model_filename).model()
        self.total_frames = self.motion.num_frames
        self.fps = self.motion.fps
    
        # camera options
        self.focus_on_root = False
        self.follow_root = False
        self.init_cam_pos = self.camera.position
    
    def start(self):
        super().start()

        # render options
        self.render_model = agl.Render.model(self.model)

        # UI options
        self.ui.add_menu("MotionApp")
        self.ui.add_menu_item("MotionApp", "Model", self.render_model.switch_visible, key=glfw.KEY_M)

    def update(self):
        super().update()

        curr_frame = self.frame % self.total_frames
        self.model.set_pose(self.motion.poses[curr_frame])

        # set camera focus on the root
        if self.focus_on_root:
            self.camera.set_focus_position(self.motion.poses[curr_frame].root_p)
            self.camera.set_up(glm.vec3(0, 1, 0))
        elif self.follow_root:
            self.camera.set_position(self.motion.poses[curr_frame].root_p + glm.vec3(0, 1.5, 5))
            self.camera.set_focus_position(self.motion.poses[curr_frame].root_p)
            self.camera.set_up(glm.vec3(0, 1, 0))
        self.camera.update()
        
    def render(self):
        super().render()
        self.render_model.update_model(self.model).draw()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        # set camera focus on the root
        if key == glfw.KEY_F3 and action == glfw.PRESS:
            self.focus_on_root = not self.focus_on_root
        elif key == glfw.KEY_F4 and action == glfw.PRESS:
            self.follow_root = not self.follow_root
            
if __name__ == "__main__":
    motion_filename = os.path.join(agl.AGL_PATH, "data/fbx/motion/ybot_walking.fbx")
    model_filename  = os.path.join(agl.AGL_PATH, "data/fbx/model/ybot.fbx")
    agl.AppManager.start(MotionApp(motion_filename, model_filename))