import torch
import glfw
import glm

import numpy as np

from pymovis.motion.data import bvh, fbx
from pymovis.motion.core import Motion

from pymovis.vis.render import Render
from pymovis.vis.app import App, MotionApp
from pymovis.vis.appmanager import AppManager
from pymovis.vis.model import Model
from pymovis.vis.heightmap import Heightmap

class SimpleApp(App):
    def __init__(self):
        super().__init__()
        self.m = Render.plane()
    
    def render(self):
        self.m.draw()

class MyApp(MotionApp):
    def __init__(self, motion: Motion, model, vel_factor):
        super().__init__(motion, model)
        self.vel_factor = vel_factor
        # self.scaled = [self.motion.scaled_motion(v) for v in self.vel_factor]

        # for idx, scaled in enumerate(self.scaled):
        #     if scaled is None:
        #         continue

            # for pose in scaled.poses:
            #     pose.translate_root_p(np.array([idx - 2, 0, 0]))

        # heightmap
        self.heightmap = Heightmap.load_from_file("./data/heightmaps/hmap_010_smooth.txt")
        self.heightmap_mesh = Render.vao(self.heightmap.vao).set_texture("grid.png").set_uv_repeat(0.1)

        # grid for environment map
        grid_x = np.linspace(-1, 1, 11)
        grid_z = np.linspace(-1, 1, 11)
        grid_x, grid_z = np.meshgrid(grid_x, grid_z)
        grid_y = np.zeros_like(grid_x)
        self.env_map = np.stack([grid_x, grid_y, grid_z], axis=-1)
        self.sphere = Render.sphere(0.05).set_albedo([0, 1, 0])
        # self.cubemap = Render.cubemap("skybox")

        self.color_dict = {
            0: [1, 0.2, 0.2],
            1: [0.2, 1, 0.2],
            2: [0.2, 0.2, 1],
            3: [1, 1, 0.2],
            4: [1, 0.2, 1],
            5: [0.2, 1, 1],
            6: [1, 1, 1],
        }

        self.text_on_screen = Render.text_on_screen("HI")

    def render(self):
        super().render()
        self.text_on_screen.draw()
        self.heightmap_mesh.draw()
        r = np.stack([self.motion.poses[self.frame].left, self.motion.poses[self.frame].up, self.motion.poses[self.frame].forward], axis=-1)
        env_map = np.einsum("ij,abj->abi", r, self.env_map) + self.motion.poses[self.frame].base
        env_map = np.reshape(env_map, [-1, 3])
        env_map[..., 1] = self.heightmap.sample_height(env_map[..., 0], env_map[..., 2])
        for e in env_map:
            self.sphere.set_position(e).draw()
            
        # for idx, scaled in enumerate(self.scaled):
        #     if scaled is not None:
        #         self.model.set_pose_by_source(scaled.poses[self.frame])
        #         Render.model(self.model).draw()
        #         # self.render_xray(scaled.poses[self.frame], self.color_dict[idx])

        #         # left = scaled.poses[self.frame].left
        #         # up = scaled.poses[self.frame].up
        #         # forward = scaled.poses[self.frame].forward
        #         # orientation = glm.mat3(*np.stack([left, up, forward], axis=-1).T.ravel())
        #         Render.text(self.vel_factor[idx]).set_position(scaled.poses[self.frame].root_p + np.array([-0.3, 1, 0])).set_scale(0.5).draw()


if __name__ == "__main__":
    app_manager = AppManager()

    motion = bvh.load("D:/data/LaFAN1/walk1_subject5.bvh", v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01)
    model = fbx.FBX("./data/models/model_skeleton.fbx").model()
    # motion.align_by_frame(0)
    motion = motion.make_window(560, 1000)
    app = MyApp(motion, model, [0.8, 0.9, 1.0, 1.1, 1.2])
    app_manager.run(app)