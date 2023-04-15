import numpy as np
from OpenGL.GL import *

import torch
from pymovis.motion import BVH, FBX, Motion
from pymovis.vis import Render, MotionApp, AppManager
from pymovis.ops import motionops, rotation

class MyApp(MotionApp):
    def __init__(self, motion, model):
        super().__init__(motion, model)
        self.motion = motion
        self.model = model

        feet_names = ["LeftFoot", "LeftToe", "RightFoot", "RightToe"]
        feet_ids = [self.motion.skeleton.idx_by_name[name] for name in feet_names]
        self.feet_ids = feet_ids

        global_ps = np.stack([pose.global_p for pose in self.motion.poses], axis=0)
        global_vs = global_ps[1:] - global_ps[:-1]
        global_vs = np.concatenate([global_vs[0:1], global_vs], axis=0)
        global_vs = np.sum(global_vs**2, axis=-1)
        self.feet_vs = global_vs[:, feet_ids]
        self.sphere = Render.sphere(0.05).set_albedo([1, 0, 0])
        
    def render(self):
        super().render(render_xray=False, model_background=0.1)

        for i, foot_id in enumerate(self.feet_ids):
            if self.feet_vs[self.frame, i] < 2e-4:
                foot_p = self.motion.poses[self.frame].global_p[foot_id]
                self.sphere.set_position(foot_p).draw()

if __name__ == "__main__":
    # app cycle manager
    app_manager = AppManager()

    # load data
    motion = BVH.load("data/motion.bvh", v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01)
    model = FBX("data/character.fbx").model()

    motion.align_to_origin_by_frame(0, axes="xz")

    # create app
    app = MyApp(motion, model)

    # run app
    app_manager.run(app)