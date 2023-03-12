from pymovis.motion import BVH, FBX
from pymovis.vis import Render, MotionApp, AppManager
from pymovis.ops import motionops
from pymovis.vis.const import YBOT_FBX_DICT

class MyApp(MotionApp):
    def __init__(self, motion, model):
        super().__init__(motion, model, YBOT_FBX_DICT)
        self.motion = motion
        self.model = model

    def render(self):
        # super().render()
        # super().render(render_model=False)
        super().render(render_xray=False)
        Render.model(self.model).draw()

if __name__ == "__main__":
    # app cycle manager
    app_manager = AppManager()

    # load data
    motion = BVH.load("data/motion2.bvh", v_forward=[0, 0, 1], v_up=[0, 1, 0], to_meter=0.01)
    # motion = BVH.load("data/bvh3/lafan_Moonwalk_mcp.bvh", v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01) # for LaFAN1 model
    model = FBX("data/ybot.fbx").model()

    # align and slice
    motion.align_by_frame(0, origin_axes="xz")
    motion = motion.make_window(0, 1000)

    # create app
    app = MyApp(motion, model)

    # run app
    app_manager.run(app)