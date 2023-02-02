from pymovis.motion.data import bvh, fbx
from pymovis.vis.app import MotionApp
from pymovis.vis.appmanager import AppManager

class MyApp(MotionApp):
    def __init__(self, motion, model):
        super().__init__(motion, model)
        self.motion = motion
        self.model = model

    def render(self):
        super().render()
        # super().render(render_model=False)
        # super().render(render_xray=False)

if __name__ == "__main__":
    # app cycle manager
    app_manager = AppManager()

    # load data
    motion = bvh.load("data/animations/flat/PFNN_LocomotionFlat01_000.bvh", v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01)
    model = fbx.FBX("data/models/model_skeleton.fbx").model()

    # align and slice
    motion.align_by_frame(0)
    motion = motion.make_window(0, 1000)

    # create app
    app = MyApp(motion, model)

    # run app
    app_manager.run(app)