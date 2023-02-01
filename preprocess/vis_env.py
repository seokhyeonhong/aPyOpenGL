import sys
sys.path.append(".")
sys.path.append("..")

import os
import pickle
import numpy as np

from OpenGL.GL import *

from pymovis.motion.data.fbx import FBX

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MotionApp
from pymovis.vis.render import Render
from pymovis.vis.heightmap import Heightmap
from pymovis.utils.config import DatasetConfig

""" Load data """
def load_windows(split, dir, filename):
    with open(os.path.join(dir, f"{split}_{filename}"), "rb") as f:
        dict = pickle.load(f)
    return dict["windows"]

def load_patches(split, dir, filename):
    with open(os.path.join(dir, f"{split}_{filename}"), "rb") as f:
        env = pickle.load(f)
    return env["patches"], env["env"]

""" Visualize """
class MyApp(MotionApp):
    def __init__(self, motion, model, patch, env, h_scale, v_scale):
        super().__init__(motion, model)
        self.motion  = motion
        self.model   = model
        self.patch   = patch
        self.env     = env
        self.h_scale = h_scale
        self.v_scale = v_scale

        # visibility settings
        self.grid.set_visible(False)
        self.axis.set_visible(False)
        self.text.set_visible(False)
        
        # foot joint indices for contact visualization
        jid_left_foot  = self.motion.skeleton.idx_by_name["LeftFoot"]
        jid_left_toe   = self.motion.skeleton.idx_by_name["LeftToe"]
        jid_right_foot = self.motion.skeleton.idx_by_name["RightFoot"]
        jid_right_toe  = self.motion.skeleton.idx_by_name["RightToe"]
        self.jid       = [jid_left_foot, jid_left_toe, jid_right_foot, jid_right_toe]
        self.contact_sphere = Render.sphere(0.05).set_albedo([1, 0, 0])

        # heightmap
        self.patch = patch
        self.patch_vao = Render.vao(Heightmap(patch, h_scale=h_scale, v_scale=v_scale, offset=0).vao).set_texture("grid.png").set_uv_repeat(0.1)

        # sensor
        sensor_x, sensor_z = np.meshgrid(np.linspace(-1, 1, 11, dtype=np.float32), np.linspace(-1, 1, 11, dtype=np.float32))
        sensor_y = np.zeros_like(sensor_x)
        self.sensor = np.stack([sensor_x, sensor_y, sensor_z], axis=-1).reshape(-1, 3)
        self.sensor_sphere = Render.sphere(0.05).set_albedo([0, 1, 0])
    
    def render(self):
        super().render()
        self.patch_vao.draw()

        # contact
        if self.frame > 0:
            curr_ps = self.motion.poses[self.frame].global_p[self.jid]
            prev_ps = self.motion.poses[self.frame-1].global_p[self.jid]
            global_v = np.sum((curr_ps - prev_ps) ** 2, axis=-1)
            contact = global_v < 2e-4 # velocity threshold
            for i in range(len(self.jid)):
                if contact[i]:
                    self.contact_sphere.set_position(curr_ps[i]).draw()
        
        # grid
        # R = np.array([self.motion.poses[self.frame].left, self.motion.poses[self.frame].up, self.motion.poses[self.frame].forward], dtype=np.float32)
        R = np.stack([self.motion.poses[self.frame].left, self.motion.poses[self.frame].up, self.motion.poses[self.frame].forward], axis=-2)
        t = self.motion.poses[self.frame].base
        grid = np.matmul(R, self.sensor.T).T + t
        grid[..., 1] = Heightmap.sample_height(self.patch, grid[..., 0], grid[..., 2], self.h_scale, self.v_scale)
        # for idx, grid_point in enumerate(grid):
        #     self.sensor_sphere.set_position(*grid_point).set_albedo([0, 1, 0]).draw()
        
        env = self.env[self.frame, 6:]
        grid[..., 1] = env
        for idx, grid_point in enumerate(grid):
            self.sensor_sphere.set_position(*grid_point).set_albedo([1, 0, 0]).draw()

def main():
    # config
    config = DatasetConfig.load("configs/config.json")
    
    # load data
    # train_windows = load_windows("train", config.dataset_dir, config.motion_pklname)
    # train_patches = load_patches("train", config.dataset_dir, config.env_pklname)
    test_windows  = load_windows("test", config.dataset_dir, config.motion_pklname)
    test_patches, test_env  = load_patches("test", config.dataset_dir, config.env_pklname)

    fbx = FBX("./data/models/model_skeleton.fbx")

    # visualize
    # for window in train_windows:
    #     for patch in train_patches:
    #         app_manager = AppManager()
    #         model = fbx.model()
    #         app = MyApp(window, model, patch, h_scale=config.h_scale, v_scale=config.v_scale)
    #         app_manager.run(app)
    
    for window in test_windows:
        for patch, env in zip(test_patches, test_env):
            app_manager = AppManager()
            model = fbx.model()
            app = MyApp(window, model, patch, env, h_scale=config.h_scale, v_scale=config.v_scale)
            app_manager.run(app)

if __name__ == "__main__":
    main()