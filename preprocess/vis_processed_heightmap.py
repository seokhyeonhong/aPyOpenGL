import sys
sys.path.append(".")

import os
import numpy as np

from pymovis.vis.heightmap import Heightmap
from pymovis.vis.app import App
from pymovis.vis.appmanager import AppManager
from pymovis.vis.render import Render
from pymovis.vis.const import INCH_TO_METER

""" Global variables """
SPARSITY      = 15
SIZE          = 200
HEIGHTMAP_DIR = "./data/heightmaps"
SAVE_DIR      = "./data/dataset/heightmap"

""" Load from saved files """
def load_all_patches():
    data = np.load(os.path.join(SAVE_DIR, f"sparsity{SPARSITY}_size{SIZE}.npy"))
    print(f"Loaded patches: {data.shape}")
    return data

""" Main function """
class MyApp(App):
    def __init__(self, heightmap):
        super().__init__()
        self.heightmap = Render.mesh(heightmap.mesh).set_texture("grid.png").set_uv_repeat(0.1)
        self.axis = Render.axis()
    
    def render(self):
        super().render()
        self.heightmap.draw()
        self.axis.draw()

def main():
    X = load_all_patches()
    for x in X:
        app_manager = AppManager()
        heightmap = Heightmap(x, h_scale=INCH_TO_METER * 2)
        app = MyApp(heightmap)
        app_manager.run(app)

if __name__ == "__main__":
    main()