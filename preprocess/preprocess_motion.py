import sys
sys.path.append(".")

import os
import pickle

import numpy as np

from pymovis.motion.data import bvh
from pymovis.motion.ops import npmotion

from pymovis.utils import util

""" Global variables for the dataset """
WINDOW_SIZE   = 50
WINDOW_OFFSET = 20
FPS           = 30
MOTION_DIR    = "./data/animations"
SAVE_DIR      = "./data/dataset/motion"
SAVE_FILENAME = f"size{WINDOW_SIZE}_offset{WINDOW_OFFSET}_fps{FPS}.npy"

""" Load from saved files """
def load_motions():
    files = []
    for file in sorted(os.listdir(MOTION_DIR)):
        if file.endswith(".bvh"):
            files.append(os.path.join(MOTION_DIR, file))
    
    motions = bvh.load_parallel(files, v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01)
    return motions

""" Save processed data """
def save_skeleton(skeleton):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    save_path = os.path.join(SAVE_DIR, "skeleton.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(skeleton, f)

def save_windows(motions):
    # load and permute the order
    windows = []
    for w in util.run_parallel_sync(get_windows, motions, desc="Extracting windows"):
        windows.extend(w)
    windows = np.stack(windows, axis=0).astype(np.float32)
    np.random.shuffle(windows)

    # split train/test
    train_windows = windows[:int(len(windows) * 0.8)]
    test_windows  = windows[int(len(windows) * 0.8):]

    print("Train windows:", train_windows.shape)
    print("Test windows:", test_windows.shape)

    # save
    print("Saving windows")
    np.save(os.path.join(SAVE_DIR, f"train_{SAVE_FILENAME}"), train_windows)
    np.save(os.path.join(SAVE_DIR, f"test_{SAVE_FILENAME}"), test_windows)

""" Extract windows a motion clip """
def get_windows(motion):
    windows = []
    for start in range(0, motion.num_frames - WINDOW_SIZE, WINDOW_OFFSET):
        window = motion.make_window(start, start + WINDOW_SIZE)

        # -------------- Modify this part to save other features --------------
        # Features to save. Modify this part to save other features.
        # Dimensions: (WINDOW_SIZE, D)
        window.align_by_frame(9)
        
        local_R = np.stack([pose.local_R for pose in window.poses], axis=0)
        root_p  = np.stack([pose.root_p for pose in window.poses], axis=0)

        local_R6 = npmotion.R_to_R6(local_R).reshape(WINDOW_SIZE, -1)
        root_p   = root_p.reshape(WINDOW_SIZE, -1)
 
        window   = np.concatenate([local_R6, root_p], axis=-1)
        windows.append(window)
        # ---------------------------------------------------------------------

    return windows

""" Main function """
def main():
    util.seed()
    motions = load_motions()
    save_skeleton(motions[0].skeleton)
    save_windows(motions)

if __name__ == "__main__":
    main()