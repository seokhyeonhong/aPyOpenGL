import sys
sys.path.append(".")

import os
import pickle

from OpenGL.GL import *

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MotionApp
from pymovis.vis.render import Render
from pymovis.vis.heightmap import Heightmap
from pymovis.vis.glconst import INCH_TO_METER

""" Global variables for the dataset """
DATASET_DIR   = "./data/dataset"

MOTION_DIR    = f"{DATASET_DIR}/motion"
WINDOW_SIZE   = 50
WINDOW_OFFSET = 20
FPS           = 30

HEIGHTMAP_DIR = f"{DATASET_DIR}/heightmap"
SPARSITY      = 15
SIZE          = 200
TOP_K_SAMPLES = 10
H_SCALE       = 2 * INCH_TO_METER
V_SCALE       = INCH_TO_METER

""" Load processed data """
def load_processed_envmap(split):
    vis_dir = os.path.join(DATASET_DIR, "vis")
    with open(os.path.join(vis_dir, f"{split}_size{WINDOW_SIZE}_offset{WINDOW_OFFSET}_sparsity{SPARSITY}_size{SIZE}_top{TOP_K_SAMPLES}.pkl"), "rb") as f:
        vis_data = pickle.load(f)

    return vis_data

""" Main functions """
def visualize(train=True, test=True):
    if train:
        train_data = load_processed_envmap("train")
        for motion, patch, envmap, contact in train_data:
            for p, e in zip(patch, envmap):
                app_manager = AppManager()
                app = MyApp(motion, contact, p, e)
                app_manager.run(app)
    
    if test:
        test_data = load_processed_envmap("test")
        for motion, patch, envmap, contact in test_data:
            for p, e in zip(patch, envmap):
                app_manager = AppManager()
                app = MyApp(motion, contact, p, e)
                app_manager.run(app)

class MyApp(MotionApp):
    def __init__(self, motion, contact, heightmap, envmap):
        super().__init__(motion)
        self.contact = contact
        self.sphere = Render.sphere().set_material([0, 1, 0])
        self.grid.set_visible(False)
        self.axis.set_visible(False)
        
        jid_left_foot  = self.motion.skeleton.idx_by_name["LeftFoot"]
        jid_left_toe   = self.motion.skeleton.idx_by_name["LeftToe"]
        jid_right_foot = self.motion.skeleton.idx_by_name["RightFoot"]
        jid_right_toe  = self.motion.skeleton.idx_by_name["RightToe"]
        self.jid       = [jid_left_foot, jid_left_toe, jid_right_foot, jid_right_toe]

        self.heightmap_mesh = Render.mesh(Heightmap(heightmap, h_scale=H_SCALE, v_scale=V_SCALE, offset=0).mesh).set_texture("grid.png").set_uv_repeat(0.1)

        self.envmap = envmap
    
    def render(self):
        super().render()
        contact = self.contact[self.frame]
        envmap = self.envmap[self.frame]

        glDisable(GL_DEPTH_TEST)
        for idx, jid in enumerate(self.jid):
            self.sphere.set_position(self.motion.global_p[self.frame, jid]).set_material([0, 1, 0]).set_scale(0.1 * contact[idx]).draw()
        glEnable(GL_DEPTH_TEST)

        self.heightmap_mesh.draw()

        for p in envmap:
            self.sphere.set_position(p).set_scale(0.1).set_material([1, 1, 0]).draw()

def main():
    visualize()

if __name__ == "__main__":
    main()