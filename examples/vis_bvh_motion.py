import os

from pymovis import AppManager, MotionApp, BVH

class MyApp(MotionApp):
    def __init__(self, motion):
        super().__init__(motion)
    
    def render(self):
        super().render(render_xray=True)

if __name__ == "__main__":
    # app cycle manager
    app_manager = AppManager()

    # load data
    filepath = os.path.join(os.path.dirname(__file__), "../data/bvh/ybot_capoeira.bvh")
    motion = BVH.load(filepath, v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01)

    # align and slice
    motion.align_by_frame(0, origin_axes="xz")

    # create app
    app = MyApp(motion)

    # run app
    app_manager.run(app)