import sys
sys.path.append(".")
sys.path.append("..")

import os
import pickle

from OpenGL.GL import *

from pymovis.motion.data.fbx import FBX

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MotionApp
from pymovis.vis.render import Render
from pymovis.vis.heightmap import Heightmap
from pymovis.vis.const import INCH_TO_METER
from pymovis.utils.config import DatasetConfig

""" Load processed data """
def load_processed_envmap(split, dir, filename):
    with open(os.path.join(dir, f"{split}_{filename}"), "rb") as f:
        vis_data = pickle.load(f)
    return vis_data

""" Visualize """
class MyApp(MotionApp):
    def __init__(self, motion, model, contact, heightmap, envmap, h_scale, v_scale):
        super().__init__(motion, model)
        self.contact = contact
        self.sphere = Render.sphere().set_albedo([0, 1, 0])
        self.grid.set_visible(False)
        self.axis.set_visible(False)
        self.text.set_visible(False)
        
        jid_left_foot  = self.motion.skeleton.idx_by_name["LeftFoot"]
        jid_left_toe   = self.motion.skeleton.idx_by_name["LeftToe"]
        jid_right_foot = self.motion.skeleton.idx_by_name["RightFoot"]
        jid_right_toe  = self.motion.skeleton.idx_by_name["RightToe"]
        self.jid       = [jid_left_foot, jid_left_toe, jid_right_foot, jid_right_toe]

        self.heightmap_mesh = Render.vao(Heightmap(heightmap, h_scale=h_scale, v_scale=v_scale, offset=0).vao).set_texture("grid.png").set_uv_repeat(0.1)

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
    # config
    config = DatasetConfig.load("configs/config.json")
    
    # load data
    train_data = load_processed_envmap("train", config.dataset_dir, config.vis_pklname)
    test_data = load_processed_envmap("test", config.dataset_dir, config.vis_pklname)
    fbx = FBX("./data/models/model_skeleton.fbx")

    # visualize
    for motion, patch, edit, envmap, contact in train_data:
        for p, e in zip(patch, envmap):
            app_manager = AppManager()
            model = fbx.model()
            app = MyApp(motion, model, contact, p, e, h_scale=config.h_scale, v_scale=config.v_scale)
            # print(motion.name, motion.type)
            app_manager.run(app)
        
    for motion, patch, edit, envmap, contact in test_data:
        for p, e in zip(patch, envmap):
            app_manager = AppManager()
            model = fbx.model()
            app = MyApp(motion, model, contact, p, e, h_scale=config.h_scale, v_scale=config.v_scale)
            app_manager.run(app)
    

if __name__ == "__main__":
    main()