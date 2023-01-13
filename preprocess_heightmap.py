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
SPARSITY      = 16
SIZE          = 200

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
        print(f"Extracting patches {idx + 1} / {len(files)}", end="\r")
        H = np.loadtxt(f)
        num_samples = (H.shape[0] * H.shape[1]) // (SPARSITY * SPARSITY)
        if H.shape[0] < SIZE or H.shape[1] < SIZE:
            continue

        patches = util.run_parallel(sample_patch, [H] * num_samples)
        patches = [p for p in patches if p is not None]
        X.extend(patches)
    
    print(f"Extracted patches: {len(X)} in {time.perf_counter() - start:.2f} seconds")
    X = np.array(X, dtype=np.float32)
    
    return X

def sample_patch(heightmap):
    """ Random location / rotation """
    xi, yi = np.random.randint(-SIZE, heightmap.shape[0] - SIZE), np.random.randint(-SIZE, heightmap.shape[1] - SIZE)
    r = np.degrees(np.random.uniform(-np.pi / 2, np.pi / 2))

    """ Sample patch """
    S = ndimage.interpolation.shift(heightmap, (xi, yi), mode="reflect")[:SIZE*2, :SIZE*2]
    S = S[::-1, :] if np.random.uniform() > 0.5 else S
    S = S[:, ::-1] if np.random.uniform() > 0.5 else S
    S = ndimage.interpolation.rotate(S, r, reshape=False, mode="reflect")

    """ Extract patch area """
    P = S[SIZE//2:SIZE//2+SIZE, SIZE//2:SIZE//2+SIZE]

    """ Subtract mean """
    P -= P.mean()

    """ Discard if height difference is too high """
    if np.any(np.abs(P) > 50):
        return

    return P

""" Main functions """
def preprocess():
    util.seed()
    heightmap_files = load_all_heightmaps()
    X = sample_all_patches(heightmap_files)
    np.savez_compressed(os.path.join(HEIGHTMAP_DIR, "heightmaps.npz"), X=X)

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
    # preprocess()
    visualize()

if __name__ == "__main__":
    main()