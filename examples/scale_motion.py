import copy

from pymovis.motion.data import bvh, fbx
from pymovis.vis.render import Render
from pymovis.vis.app import MotionApp
from pymovis.vis.appmanager import AppManager
from pymovis.ops import motionops

class MyApp(MotionApp):
    def __init__(self, motion, model, vel_factor):
        super().__init__(motion, model)
        self.vel_factor = vel_factor
        self.scaled = [motionops.scaled_motion(motion, v) for v in self.vel_factor]
        self.copy_model = copy.deepcopy(self.model)

        for idx, scaled in enumerate(self.scaled):
            if scaled is None:
                continue

    def render(self):
        super().render(render_xray=False)
        for idx, scaled in enumerate(self.scaled):
            if scaled is None or self.vel_factor[idx] == 1.0:
                continue

            self.copy_model.set_pose_by_source(scaled.poses[self.frame])
            Render.model(self.copy_model).set_albedo([0, 0.5, 0], 3).draw()

if __name__ == "__main__":
    # app cycle manager
    app_manager = AppManager()

    # load data
    motion = bvh.load("data/motion2.bvh", v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01)
    model = fbx.FBX("data/character.fbx").model()

    # align and split
    motion.align_by_frame(100)
    motion = motion.make_window(100, 1000)

    # create app for scaled motion
    app = MyApp(motion, model, [0.8, 0.9, 1.0, 1.1, 1.2])

    # run app
    app_manager.run(app)