import sys
sys.path.append(".")
sys.path.append("..")

import os
import pickle
import numpy as np

from pymovis.motion.data import bvh
from pymovis.motion.ops import npmotion
from pymovis.utils import util
from pymovis.utils.config import DatasetConfig

""" Load from saved files """
def load_motions(load_dir):
    files = []
    dirs = []
    for dir in sorted(os.listdir(load_dir)):
        if dir not in ["flat", "jumpy", "rocky", "beam"]:
            continue
        for file in sorted(os.listdir(os.path.join(load_dir, dir))):
            if file.endswith(".bvh"):
                files.append(os.path.join(load_dir, dir, file))
                dirs.append(dir)
    
    motions = util.run_parallel_sync(bvh.load_with_type, zip(files, dirs), v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01, desc=f"Loading {len(files)} motions")
    return motions

""" Save processed data """
def save_skeleton(skeleton, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, "skeleton.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(skeleton, f)

def save_windows(motions, window_length, window_offset, save_dir, save_filename):
    # extract windows and features
    windows, features = [], []
    for w, f in util.run_parallel_sync(windows_and_features, motions, window_length=window_length, window_offset=window_offset, desc="Extracting windows and features"):
        windows.extend(w)
        features.append(f)
    features = np.concatenate(features, axis=0)

    # permute the order and split train/test data (80% / 20%)
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
    with open(os.path.join(save_dir, f"train_{save_filename}"), "wb") as f:
        pickle.dump(train_dict, f)
    with open(os.path.join(save_dir, f"test_{save_filename}"), "wb") as f:
        pickle.dump(test_dict, f)

""" Preprocessing function """
def windows_and_features(motion, window_length, window_offset):
    windows = []
    local_R6s, root_ps = [], []

    for start in range(0, motion.num_frames - window_length, window_offset):
        window = motion.make_window(start, start + window_length)
        window.align_by_frame(9)

        local_R = np.stack([pose.local_R for pose in window.poses], axis=0)
        root_p  = np.stack([pose.root_p for pose in window.poses], axis=0)

        local_R6 = npmotion.R_to_R6(local_R).reshape(window_length, -1)
        root_p   = root_p.reshape(window_length, -1)

        local_R6s.append(local_R6)
        root_ps.append(root_p)
        windows.append(window)
    
    local_R6s = np.stack(local_R6s, axis=0)
    root_ps   = np.stack(root_ps, axis=0)
    features  = np.concatenate([local_R6s, root_ps], axis=-1)

    return windows, features

def main():
    # config
    config = DatasetConfig.load("configs/config.json")

    # load
    util.seed()
    motions = load_motions(config.motion_dir)

    # save
    save_skeleton(motions[0].skeleton, config.dataset_dir)
    save_windows(motions, config.window_length, config.window_offset, config.dataset_dir, config.motion_pklname)

if __name__ == "__main__":
    main()