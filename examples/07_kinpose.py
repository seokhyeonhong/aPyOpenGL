import os
import glm
import glfw
import numpy as np

from pymovis import vis, kin

class MotionApp(vis.AnimApp):
    def __init__(self, motion_filename, model_filename):
        super().__init__()

        # motion data
        self.motion = vis.FBX(motion_filename).motions()[0]
        self.model  = vis.FBX(model_filename).model()
        self.total_frames = self.motion.num_frames()
        self.fps = self.motion.fps
    
    def start(self):
        super().start()

        # render options
        self.render_skeleton = vis.Render.skeleton(self.model)
        self.render_model = vis.Render.model(self.model)

        # kin pose
        self.kinpose = kin.KinPose(self.motion.poses[0])

        # UI options
        self.ui.add_menu("MotionApp")
        self.ui.add_menu_item("MotionApp", "X-Ray", self.render_skeleton.switch_visible, key=glfw.KEY_X)
        self.ui.add_menu_item("MotionApp", "Model", self.render_model.switch_visible, key=glfw.KEY_M)

    def update(self):
        super().update()
        
        # update kinpose basis to the origin
        self.kinpose.set_pose(self.motion.poses[self.curr_frame])
        self.kinpose.set_basis_xform(np.eye(4))

        # update model to render
        self.model.set_pose(self.kinpose.to_pose())

    def render(self):
        super().render()
        self.render_model.update_model(self.model).draw()

    def render_xray(self):
        super().render_xray()
        self.render_skeleton.update_skeleton(self.model).draw()

if __name__ == "__main__":
    motion_filename = os.path.join(os.path.dirname(__file__), "../data/fbx/motion/ybot_walking.fbx")
    model_filename  = os.path.join(os.path.dirname(__file__), "../data/fbx/model/ybot.fbx")
    vis.AppManager.start(MotionApp(motion_filename, model_filename))