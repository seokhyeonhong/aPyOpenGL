import sys
sys.path.append(".")
sys.path.append("..")

import os
import pickle

from pymovis.vis.heightmap import Heightmap
from pymovis.vis.app import App
from pymovis.vis.appmanager import AppManager
from pymovis.vis.render import Render
from pymovis.vis.const import INCH_TO_METER
from pymovis.utils.config import DatasetConfig

""" Load from saved files """
def load_patches(save_dir, save_filename):
    with open(os.path.join(save_dir, save_filename), "rb") as f:
        data = pickle.load(f)
    print(f"Loaded patches: {data.shape}")
    return data

""" Main function """
class MyApp(App):
    def __init__(self, heightmap: Heightmap):
        super().__init__()
        self.heightmap = Render.vao(heightmap.vao).set_texture("grid.png").set_uv_repeat(0.1)
        self.axis = Render.axis()
    
    def render(self):
        super().render()
        self.heightmap.draw()
        self.axis.draw()

def main():
    # config
    config = DatasetConfig.load("configs/config.json")

    # load data
    patches = load_patches(config.dataset_dir, config.heightmap_pklname)

    # visualize
    for patch in patches:
        app_manager = AppManager()
        heightmap = Heightmap(patch, h_scale=config.h_scale, v_scale=config.v_scale)
        app = MyApp(heightmap)
        app_manager.run(app)

if __name__ == "__main__":
    main()