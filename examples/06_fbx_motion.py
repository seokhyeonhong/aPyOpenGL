import os
import glfw

from pymovis import agl

class MotionApp(agl.AnimApp):
    def __init__(self, motion_filename, model_filename):
        super().__init__()

        # motion data
        self.motion = agl.FBX(motion_filename).motions()[0]
        self.model  = agl.FBX(model_filename).model()
        self.total_frames = self.motion.num_frames()
        self.fps = self.motion.fps
    
    def start(self):
        super().start()

        # render options
        self.render_skeleton = agl.Render.skeleton(self.model)
        self.render_model = agl.Render.model(self.model)

        # UI options
        self.ui.add_menu("MotionApp")
        self.ui.add_menu_item("MotionApp", "X-Ray", self.render_skeleton.switch_visible, key=glfw.KEY_X)
        self.ui.add_menu_item("MotionApp", "Model", self.render_model.switch_visible, key=glfw.KEY_M)

    def update(self):
        super().update()
        self.model.set_pose(self.motion.poses[self.curr_frame])

    def render(self):
        super().render()
        self.render_model.update_model(self.model).draw()

    def render_xray(self):
        super().render_xray()
        self.render_skeleton.update_skeleton(self.model).draw()

if __name__ == "__main__":
    motion_filename = os.path.join(agl.AGL_PATH, "data/fbx/motion/ybot_capoeira.fbx")
    model_filename  = os.path.join(agl.AGL_PATH, "data/fbx/model/ybot.fbx")
    agl.AppManager.start(MotionApp(motion_filename, model_filename))