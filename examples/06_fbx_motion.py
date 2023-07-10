import os

from pymovis import vis

class MyApp(vis.MotionApp):
    def __init__(self, motion_filename, model_filename):
        super().__init__()
        self.motion = vis.FBX(motion_filename).motions()[0]
        self.model  = vis.FBX(model_filename).model()

if __name__ == "__main__":
    motion_filename = os.path.join(os.path.dirname(__file__), "../data/fbx/motion/ybot_capoeira.fbx")
    model_filename  = os.path.join(os.path.dirname(__file__), "../data/fbx/model/ybot.fbx")
    vis.AppManager.start(MyApp(motion_filename, model_filename))