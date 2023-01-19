import sys
sys.path.append(".")

import os
import time
import numpy as np
import scipy.ndimage as ndimage

from pymovis.utils import util

""" Global variables """
SPARSITY      = 15
SIZE          = 200
HEIGHTMAP_DIR = "./data/heightmaps"
SAVE_DIR      = "./data/dataset/heightmap"

""" Load from saved files """
def load_all_heightmaps():
    files = []
    for f in os.listdir(HEIGHTMAP_DIR):
        if f.endswith(".txt"):
            files.append(os.path.join(HEIGHTMAP_DIR, f))
    return files

""" Sample patches """
def sample_all_patches(files):
    X = []
    start = time.perf_counter()
    sum_samples = 0
    for idx, f in enumerate(files):
        # load heightmap
        H = np.loadtxt(f)
        num_samples = (H.shape[0] * H.shape[1]) // (SPARSITY * SPARSITY)

        # skip if terrain is too small to sample (SIZE x SIZE) patches
        if SIZE//2+SIZE >= H.shape[0] or SIZE//2+SIZE >= H.shape[1]:
            print(f"Skipping {f} due to small size")
            continue

        # random location / rotation
        x = np.random.randint(-SIZE, H.shape[0] - SIZE, size=num_samples)
        y = np.random.randint(-SIZE, H.shape[1] - SIZE, size=num_samples)
        d = np.degrees(np.random.uniform(-np.pi / 2, np.pi / 2, size=num_samples))
        flip_x = np.random.uniform(size=num_samples)
        flip_y = np.random.uniform(size=num_samples)
        
        # sample patches in parallel
        patches = util.run_parallel_sync(sample_patch, zip(x, y, d, flip_x, flip_y), heightmap=H, desc=f"Sampling {num_samples} patches [Size: {SIZE} x {SIZE}] [Terrain: {H.shape[0]} x {H.shape[1]}] [Progress: {idx + 1} / {len(files)}]")
        patches = [p for p in patches if p is not None]

        X.extend(patches)
        sum_samples += num_samples
    
    print(f"Extracted patches: {len(X)} in {time.perf_counter() - start:.2f} seconds")
    X = np.stack(X, axis=0).astype(np.float32)
    
    return X

def sample_patch(xyd_fx_fy, heightmap):
    x, y, d, flip_x, flip_y = xyd_fx_fy

    S = ndimage.interpolation.shift(heightmap, (x, y), mode="reflect")[:SIZE*2, :SIZE*2]
    S = S[::-1, :] if flip_x > 0.5 else S
    S = S[:, ::-1] if flip_y > 0.5 else S
    S = ndimage.interpolation.rotate(S, d, reshape=False, mode="reflect")

    P = S[SIZE//2:SIZE//2+SIZE, SIZE//2:SIZE//2+SIZE]
    P -= P.mean()

    return None if np.any(np.abs(P) > 50) else P

""" Main function """
def main():
    util.seed()
    
    heightmap_files = load_all_heightmaps()
    patches = sample_all_patches(heightmap_files)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    np.save(os.path.join(SAVE_DIR, f"sparsity{SPARSITY}_size{SIZE}.npy"), patches)
    print(f"Saved patches: {patches.shape}")

if __name__ == "__main__":
    main()