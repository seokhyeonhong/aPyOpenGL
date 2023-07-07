import os
import numpy as np
from OpenGL.GL import *
import glfw, glm
import copy

from pymovis import FBX, BVH, Render, MotionApp, AppManager, YBOT_FBX_DICT

class MyApp(MotionApp):
    def __init__(self, motion, model):
        super().__init__(motion, model, YBOT_FBX_DICT)
        self.motion = motion
        self.model = model

        self.show_kf = False
        self.show_trans = False
        self.show_constrained = False
        
        # keyframe
        self.kf_model = copy.deepcopy(self.model)
        self.kf_model.meshes[0].materials[0].albedo = glm.vec3(0.4, 0.2, 0.6)
        self.kf_model.meshes[1].materials[0].albedo = glm.vec3(0.1, 0.1, 0.3)
        self.kf_model.set_source_skeleton(self.motion.skeleton, YBOT_FBX_DICT)

        # transition
        self.trans_model = copy.deepcopy(self.model)
        self.trans_model.meshes[0].materials[0].albedo = glm.vec3(0.8)
        self.trans_model.meshes[1].materials[0].albedo = glm.vec3(0.2)

        # sphere
        self.show_sphere = True
        self.sphere = Render.sphere(0.03).set_albedo(glm.vec3(1, 0, 0))
        self.arrow = Render.arrow().set_albedo(glm.vec3(1, 0, 0))

    def render(self):
        super().render(render_xray=True)

        bg = False

        # first and last frame
        if self.show_constrained:
            self.model.set_pose_by_source(self.motion.poses[0])
            Render.model(self.model).draw()
            self.model.set_pose_by_source(self.motion.poses[5])
            Render.model(self.model).draw()
            self.model.set_pose_by_source(self.motion.poses[10])
            Render.model(self.model).draw()

            self.model.set_pose_by_source(self.motion.poses[-1])
            Render.model(self.model).draw()
        
        # draw by thirty frames
        if self.show_kf:
            for i in range(30, len(self.motion), 30):
                self.kf_model.set_pose_by_source(self.motion.poses[i])
                Render.model(self.kf_model).draw()
        
        # draw by ten frames
        if self.show_trans:
            for i in range(10, len(self.motion), 10):
                self.trans_model.set_pose_by_source(self.motion.poses[i])
                Render.model(self.trans_model).draw()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        if key == glfw.KEY_Q and action == glfw.PRESS:
            self.show_kf = not self.show_kf
        if key == glfw.KEY_W and action == glfw.PRESS:
            self.show_trans = not self.show_trans
        if key == glfw.KEY_E and action == glfw.PRESS:
            self.show_constrained = not self.show_constrained
        if key == glfw.KEY_A and action == glfw.PRESS:
            self.show_sphere = not self.show_sphere

    def render_text(self):
        super().render_text()
        
if __name__ == "__main__":
    # app cycle manager
    app_manager = AppManager()

    # load data
    motion_path = os.path.join(os.path.dirname(__file__), "../data/bvh/ybot_capoeira.bvh")
    model_path  = os.path.join(os.path.dirname(__file__), "../data/fbx/model/ybot.fbx")

    motion = BVH.load(motion_path, to_meter=0.01, v_forward=[-1, 0, 0])
    model = FBX(model_path).model()

    # create app
    app = MyApp(motion, model)

    # run app
    app_manager.run(app)