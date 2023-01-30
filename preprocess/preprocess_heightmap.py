import sys
sys.path.append(".")
sys.path.append("..")

import os
import pickle
import time
import numpy as np
import scipy.ndimage as ndimage

from pymovis.utils import util
from pymovis.utils.config import DatasetConfig

""" Load from saved files """
def load_all_heightmaps(load_dir):
    files = []
    for f in sorted(os.listdir(load_dir)):
        if f.endswith(".txt"):
            files.append(os.path.join(load_dir, f))
    return files

""" Preprocessing functions """
def sample_all_patches(files, sparsity, mapsize):
    X = []
    start = time.perf_counter()
    sum_samples = 0
    for idx, f in enumerate(files):
        # load heightmap
        H = np.loadtxt(f)
        num_samples = (H.shape[0] * H.shape[1]) // (sparsity * sparsity)

        # skip if terrain is too small to sample (SIZE x SIZE) patches
        if mapsize//2 + mapsize >= H.shape[0] or mapsize//2 + mapsize >= H.shape[1]:
            print(f"Skipping {f} due to small size")
            continue

        # random location / rotation
        x = np.random.randint(-mapsize, H.shape[0] - mapsize, size=num_samples)
        y = np.random.randint(-mapsize, H.shape[1] - mapsize, size=num_samples)
        d = np.degrees(np.random.uniform(-np.pi / 2, np.pi / 2, size=num_samples))
        flip_x = np.random.uniform(size=num_samples)
        flip_y = np.random.uniform(size=num_samples)
        
        # sample patches in parallel
        patches = util.run_parallel_sync(sample_patch, zip(x, y, d, flip_x, flip_y), heightmap=H, mapsize=mapsize, desc=f"Sampling {num_samples} patches [Size: {mapsize} x {mapsize}] [Terrain: {H.shape[0]} x {H.shape[1]}] [Progress: {idx + 1} / {len(files)}]")
        patches = [p for p in patches if p is not None]

        X.extend(patches)
        sum_samples += num_samples
    
    print(f"Extracted patches: {len(X)} in {time.perf_counter() - start:.2f} seconds")
    X = np.stack(X, axis=0).astype(np.float32)
    
    return X

def sample_patch(xyd_fx_fy, heightmap, mapsize):
    x, y, d, flip_x, flip_y = xyd_fx_fy

    S = ndimage.interpolation.shift(heightmap, (x, y), mode="reflect")[:mapsize*2, :mapsize*2]
    S = S[::-1, :] if flip_x > 0.5 else S
    S = S[:, ::-1] if flip_y > 0.5 else S
    S = ndimage.interpolation.rotate(S, d, reshape=False, mode="reflect")

    P = S[mapsize//2:mapsize//2+mapsize, mapsize//2:mapsize//2+mapsize]
    P -= P.mean()

    return None if np.any(np.abs(P) > 50) else P

""" Main function """
def main():
    # config
    config = DatasetConfig.load("configs/config.json")
    
    # preprocess
    util.seed()
    heightmaps = load_all_heightmaps(config.heightmap_dir)
    patches    = sample_all_patches(heightmaps, config.sparsity, config.mapsize)

    # save
    with open(os.path.join(config.dataset_dir, config.heightmap_pklname), "wb") as f:
        pickle.dump(patches, f)
        
    print(f"Saved patches: {patches.shape}")

if __name__ == "__main__":
    main()