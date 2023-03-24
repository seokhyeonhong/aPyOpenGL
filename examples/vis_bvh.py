from pymovis.motion import BVH
from pymovis.vis import MotionApp, AppManager

class MyApp(MotionApp):
    def __init__(self, motion):
        super().__init__(motion)
    
    def render(self):
        super().render(render_model=False, render_xray=True)

if __name__ == "__main__":
    # app cycle manager
    app_manager = AppManager()

    # load data
    motion = BVH.load("data/motion.bvh", v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01)

    # align and slice
    motion.align_by_frame(0, origin_axes="xz")
    motion = motion.make_window(0, 1000)

    # create app
    app = MyApp(motion)

    # run app
    app_manager.run(app)