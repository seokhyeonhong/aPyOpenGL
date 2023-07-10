import os

from pymovis import vis

class MyApp(vis.MotionApp):
    def __init__(self, bvh_filename):
        super().__init__()
        self.motion = vis.BVH(bvh_filename).motion()

if __name__ == "__main__":
    bvh_filename = os.path.join(os.path.dirname(__file__), "../data/bvh/ybot_capoeira.bvh")
    vis.AppManager.start(MyApp(bvh_filename))