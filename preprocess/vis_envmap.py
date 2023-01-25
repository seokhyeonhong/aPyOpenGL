import sys
sys.path.append(".")

import os
import pickle

from OpenGL.GL import *

from pymovis.motion.data.fbx import FBX

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MotionApp
from pymovis.vis.render import Render
from pymovis.vis.heightmap import Heightmap
from pymovis.vis.const import INCH_TO_METER

""" Global variables for the dataset """
WINDOW_SIZE     = 50
WINDOW_OFFSET   = 20
FPS             = 30
MOTION_DIR      = f"./data/dataset/motion"
MOTION_FILENAME = f"size{WINDOW_SIZE}_offset{WINDOW_OFFSET}_fps{FPS}.npy"

SPARSITY        = 15
SIZE            = 200
TOP_K_SAMPLES   = 10
H_SCALE         = 2 * INCH_TO_METER
V_SCALE         = INCH_TO_METER
HEIGHTMAP_DIR   = f"./data/dataset/heightmap"
HEIGHT_FILENAME = f"sparsity{SPARSITY}_size{SIZE}.npy"

VIS_DIR         = f"./data/dataset/vis/"
SAVE_FILENAME   = f"size{WINDOW_SIZE}_offset{WINDOW_OFFSET}_fps{FPS}_sparsity{SPARSITY}_size{SIZE}_top{TOP_K_SAMPLES}.pkl"

""" Load processed data """
def load_processed_envmap(split):
    with open(os.path.join(VIS_DIR, f"{split}_{SAVE_FILENAME}"), "rb") as f:
        vis_data = pickle.load(f)

    return vis_data

""" Main functions """
def visualize(train=True, test=True):
    if train:
        train_data = load_processed_envmap("train")
        for motion, patch, envmap, contact in train_data:
            for p, e in zip(patch, envmap):
                app_manager = AppManager()
                model = FBX("./data/models/model_skeleton.fbx").model()
                app = MyApp(motion, model, contact, p, e)
                app_manager.run(app)
    
    if test:
        test_data = load_processed_envmap("test")
        for motion, patch, envmap, contact in test_data:
            for p, e in zip(patch, envmap):
                app_manager = AppManager()
                model = FBX("./data/models/model_skeleton.fbx").model()
                app = MyApp(motion, model, contact, p, e)
                app_manager.run(app)

class MyApp(MotionApp):
    def __init__(self, motion, model, contact, heightmap, envmap):
        super().__init__(motion, model)
        self.contact = contact
        self.sphere = Render.sphere().set_albedo([0, 1, 0])
        self.grid.set_visible(False)
        self.axis.set_visible(False)
        
        jid_left_foot  = self.motion.skeleton.idx_by_name["LeftFoot"]
        jid_left_toe   = self.motion.skeleton.idx_by_name["LeftToe"]
        jid_right_foot = self.motion.skeleton.idx_by_name["RightFoot"]
        jid_right_toe  = self.motion.skeleton.idx_by_name["RightToe"]
        self.jid       = [jid_left_foot, jid_left_toe, jid_right_foot, jid_right_toe]

        self.heightmap_mesh = Render.vao(Heightmap(heightmap, h_scale=H_SCALE, v_scale=V_SCALE, offset=0).vao).set_texture("grid.png").set_uv_repeat(0.1)

        self.envmap = envmap
    
    def render(self):
        super().render()
        contact = self.contact[self.frame]
        envmap = self.envmap[self.frame]

        glDisable(GL_DEPTH_TEST)
        for idx, jid in enumerate(self.jid):
            self.sphere.set_position(self.motion.poses[self.frame].global_p[jid]).set_albedo([1, 0, 0]).set_scale(0.1 * contact[idx]).draw()
        glEnable(GL_DEPTH_TEST)

        self.heightmap_mesh.draw()

        for p in envmap:
            self.sphere.set_position(p).set_scale(0.1).set_albedo([0, 1, 0]).draw()

def main():
    visualize(test=False)

if __name__ == "__main__":
    main()