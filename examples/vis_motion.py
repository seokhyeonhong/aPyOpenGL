from pymovis.motion.data import bvh
from pymovis.vis.app import MotionApp
from pymovis.vis.appmanager import AppManager

if __name__ == "__main__":
    # app cycle manager
    app_manager = AppManager()

    # load data
    motion = bvh.load("data/animations/flat/PFNN_LocomotionFlat01_000.bvh", v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01)

    # align and slice
    motion.align_by_frame(0)
    motion = motion.make_window(0, 1000)

    # create app
    app = MotionApp(motion)

    # run app
    app_manager.run(app)