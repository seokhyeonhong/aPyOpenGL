import os

from pymovis.vis import MotionApp, AppManager, BVH

class MyApp(MotionApp):
    def __init__(self, bvh_filename):
        super().__init__()
        self.motion = BVH(bvh_filename).motion()

if __name__ == "__main__":
    bvh_filename = os.path.join(os.path.dirname(__file__), "../data/bvh/ybot_capoeira.bvh")
    AppManager.start(MyApp(bvh_filename))