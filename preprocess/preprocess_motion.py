import os
import pickle

import numpy as np
from tqdm import tqdm

from pymovis.motion.data import bvh
from pymovis.motion.ops import npmotion

from pymovis.utils import util

""" Global variables for the dataset """
WINDOW_SIZE   = 50
WINDOW_OFFSET = 20
FPS           = 30
MOTION_DIR    = "../data/animations"
SAVE_DIR      = "../data/dataset/motion"
SAVE_FILENAME = f"length{WINDOW_SIZE}_offset{WINDOW_OFFSET}_fps{FPS}.pkl"

""" Load from saved files """
def load_motions():
    files = []
    dirs = []
    for dir in sorted(os.listdir(MOTION_DIR)):
        if dir not in ["flat", "jumpy", "rocky"]:
            continue
        for file in sorted(os.listdir(os.path.join(MOTION_DIR, dir))):
            if file.endswith(".bvh"):
                files.append(os.path.join(MOTION_DIR, dir, file))
                dirs.append(dir)
    
    motions = util.run_parallel_sync(bvh.load_with_type, zip(files, dirs), v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01, desc=f"Loading {len(files)} motions")
    return motions

""" Save processed data """
def save_skeleton(skeleton):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    save_path = os.path.join(SAVE_DIR, "skeleton.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(skeleton, f)

def save_windows(motions):
    # load data
    windows = []
    for motion in tqdm(motions, desc="Extracting windows"):
        for start in range(0, motion.num_frames - WINDOW_SIZE, WINDOW_OFFSET):
            window = motion.make_window(start, start + WINDOW_SIZE)
            window.align_by_frame(9)
            windows.append(window)
    
    # get features
    local_R6s, root_ps = [], []
    for window in windows:
        local_R = np.stack([pose.local_R for pose in window.poses], axis=0)
        root_p  = np.stack([pose.root_p for pose in window.poses], axis=0)

        local_R6 = npmotion.R_to_R6(local_R).reshape(WINDOW_SIZE, -1)
        root_p   = root_p.reshape(WINDOW_SIZE, -1)

        local_R6s.append(local_R6)
        root_ps.append(root_p)
    
    local_R6s = np.stack(local_R6s, axis=0)
    root_ps   = np.stack(root_ps, axis=0)
    features  = np.concatenate([local_R6s, root_ps], axis=-1)

    # permute the order and split train/test data
    perm = np.random.permutation(len(windows))
    train_dict = {
        "windows":  [windows[i] for i in perm[:int(len(windows) * 0.8)]],
        "features": features[perm[:int(len(windows) * 0.8)]]
    }
    test_dict = {
        "windows":  [windows[i] for i in perm[int(len(windows) * 0.8):]],
        "features": features[perm[int(len(windows) * 0.8):]]
    }

    print("Train features:", train_dict["features"].shape)
    print("Test features:",  test_dict["features"].shape)

    # save
    print("Saving windows and features")
    with open(os.path.join(SAVE_DIR, f"train_{SAVE_FILENAME}"), "wb") as f:
        pickle.dump(train_dict, f)
    with open(os.path.join(SAVE_DIR, f"test_{SAVE_FILENAME}"), "wb") as f:
        pickle.dump(test_dict, f)

""" Main function """
def main():
    util.seed()
    motions = load_motions()
    save_skeleton(motions[0].skeleton)
    save_windows(motions)

if __name__ == "__main__":
    main()