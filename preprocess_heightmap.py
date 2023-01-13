import os

import time
import numpy as np
import scipy.ndimage as ndimage

from pymovis.utils import util
from pymovis.vis.heightmap import Heightmap
from pymovis.vis.app import App
from pymovis.vis.appmanager import AppManager
from pymovis.vis.render import Render

""" Global variables """
HEIGHTMAP_DIR = "D:/data/PFNN/heightmaps"
SPARSITY      = 15
SIZE          = 128

""" Load from saved files """
def load_all_heightmaps():
    files = []
    for f in os.listdir(HEIGHTMAP_DIR):
        if f.endswith(".txt"):
            files.append(os.path.join(HEIGHTMAP_DIR, f))
    return files

def load_all_patches():
    data = np.load(os.path.join(HEIGHTMAP_DIR, "heightmaps.npz"))
    X = data["X"].astype(np.float32)
    print(f"Loaded patches: {X.shape}")
    return X

""" Sample patches """
def sample_all_patches(files):
    X = []
    start = time.perf_counter()
    for idx, f in enumerate(files):
        print(f"Extracting patches {idx + 1} / {len(files)} ... Elapsed time: {time.perf_counter() - start:.2f} seconds", end="\r")

        """ Load heightmap """
        H = np.loadtxt(f)
        num_samples = (H.shape[0] * H.shape[1]) // (SPARSITY * SPARSITY)
        if H.shape[0] < SIZE or H.shape[1] < SIZE:
            print(f"\nSkipping {f} due to small size")
            continue

        """ Random location / rotation """
        x = np.random.randint(-SIZE, H.shape[0] - SIZE, size=num_samples)
        y = np.random.randint(-SIZE, H.shape[1] - SIZE, size=num_samples)
        d = np.degrees(np.random.uniform(-np.pi / 2, np.pi / 2, size=num_samples))
        flip_x = np.random.uniform(size=num_samples)
        flip_y = np.random.uniform(size=num_samples)
        
        """ Sample patches """
        patches = util.run_parallel(sample_patch, zip(x, y, d, flip_x, flip_y), heightmap=H)
        patches = [p for p in patches if p is not None]

        X.extend(patches)
    
    print(f"\nExtracted patches: {len(X)} in {time.perf_counter() - start:.2f} seconds")
    X = np.array(X, dtype=np.float32)
    
    return X

def sample_patch(xydlr, heightmap):
    x, y, d, flip_x, flip_y = xydlr

    S = ndimage.interpolation.shift(heightmap, (x, y), mode="reflect")[:SIZE*2, :SIZE*2]
    S = S[::-1, :] if flip_x > 0.5 else S
    S = S[:, ::-1] if flip_y > 0.5 else S
    S = ndimage.interpolation.rotate(S, d, reshape=False, mode="reflect")

    P = S[SIZE//2:SIZE//2+SIZE, SIZE//2:SIZE//2+SIZE]

    return None if np.max(P) - np.min(P) > 100 else P - P.mean()

""" Main functions """
def preprocess():
    util.seed()
    
    heightmap_files = load_all_heightmaps()
    X = sample_all_patches(heightmap_files)

    np.savez_compressed(os.path.join(HEIGHTMAP_DIR, "heightmaps.npz"), X=X)
    print(f"Saved patches: {X.shape}")

def visualize():
    class MyApp(App):
        def __init__(self, heightmap):
            super().__init__()
            self.heightmap = Render.mesh(heightmap.mesh).set_texture("grid.png").set_uv_repeat(0.1)
            self.axis = Render.axis()
        
        def render(self):
            super().render()
            self.heightmap.draw()
            self.axis.draw()

    X = load_all_patches()
    for x in X:
        app_manager = AppManager()
        heightmap = Heightmap(x)
        app = MyApp(heightmap)
        app_manager.run(app)

def main():
    preprocess()
    visualize()

if __name__ == "__main__":
    main()