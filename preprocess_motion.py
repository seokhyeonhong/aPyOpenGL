import os
import pickle

import numpy as np

from pymovis.motion.data import bvh
from pymovis.motion.ops import npmotion
from pymovis.motion.core import Motion

from pymovis.utils import util

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MotionApp
from pymovis.vis.render import Render

""" Global variables for the dataset """
WINDOW_SIZE   = 50
WINDOW_OFFSET = 20
FPS           = 30
MOTION_DIR    = "./data/animations"

""" Load from saved files """
def load_motions():
    files = []
    for file in os.listdir(MOTION_DIR):
        if file.endswith(".bvh"):
            files.append(os.path.join(MOTION_DIR, file))
    
    motions = bvh.load_parallel(files, v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01)
    return motions

def load_processed_data():
    # skeleton
    with open(os.path.join(MOTION_DIR, "processed", "skeleton.pkl"), "rb") as f:
        skeleton = pickle.load(f)

    # windows
    train_windows = np.load(os.path.join(MOTION_DIR, "processed", f"motion_train_size{WINDOW_SIZE}_offset{WINDOW_OFFSET}_fps{FPS}.npy"))
    test_windows  = np.load(os.path.join(MOTION_DIR, "processed", f"motion_test_size{WINDOW_SIZE}_offset{WINDOW_OFFSET}_fps{FPS}.npy"))

    return skeleton, train_windows, test_windows

""" Save processed data """
def save_skeleton(skeleton):
    save_dir = os.path.join(MOTION_DIR, "processed")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, "skeleton.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(skeleton, f)

def save_windows(motions):
    save_dir = os.path.join(MOTION_DIR, "processed")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_windows, test_windows = [], []
    for w in util.run_parallel(get_windows, motions, desc="Extracting windows"):
        train_windows.extend(w[:int(len(w) * 0.8)])
        test_windows.extend(w[int(len(w) * 0.8):])

    train_windows = np.stack(train_windows, axis=0).astype(np.float32)
    test_windows  = np.stack(test_windows, axis=0).astype(np.float32)
    print("Train windows:", train_windows.shape)
    print("Test windows:", test_windows.shape)

    print("Saving windows")
    np.save(os.path.join(save_dir, f"motion_train_size{WINDOW_SIZE}_offset{WINDOW_OFFSET}_fps{FPS}.npy"), train_windows)
    np.save(os.path.join(save_dir, f"motion_test_size{WINDOW_SIZE}_offset{WINDOW_OFFSET}_fps{FPS}.npy"), test_windows)

""" Extract windows a motion clip """
def get_windows(motion):
    windows = []
    for start in range(0, motion.num_frames - WINDOW_SIZE, WINDOW_OFFSET):
        window = motion.make_window(start, start + WINDOW_SIZE)

        # -------------- Modify this part to save other features --------------
        # Features to save. Modify this part to save other features.
        # Dimensions: (WINDOW_SIZE, D)
        window.align_by_frame(9)
        
        local_R6 = npmotion.R6.from_R(window.local_R).reshape(WINDOW_SIZE, -1)
        root_p   = window.root_p.reshape(WINDOW_SIZE, -1)
 
        window   = np.concatenate([local_R6, root_p], axis=-1)
        windows.append(window)
        # ---------------------------------------------------------------------

    return windows

""" Main functions """
def preprocess():
    motions = load_motions()
    save_skeleton(motions[0].skeleton)
    save_windows(motions)

def visualize(step=1, train=True, test=True):
    skeleton, train_windows, test_windows = load_processed_data()

    if train:
        for window in train_windows[::step]:
            local_R6, root_p = window[:, :-3], window[:, -3:]
            local_R = npmotion.R.from_R6(local_R6.reshape(-1, 6)).reshape(WINDOW_SIZE, -1, 3, 3)
            motion = Motion.from_numpy(skeleton, local_R, root_p, fps=FPS)

            app_manager = AppManager()
            app = MotionApp(motion)
            app_manager.run(app)
    
    if test:
        for window in test_windows[::step]:
            local_R6, root_p = window[:, :-3], window[:, -3:]
            local_R = npmotion.R.from_R6(local_R6.reshape(-1, 6)).reshape(WINDOW_SIZE, -1, 3, 3)
            motion = Motion.from_numpy(skeleton, local_R, root_p, fps=FPS)

            app_manager = AppManager()
            app = MotionApp(motion)
            app_manager.run(app)

def main():
    preprocess()
    visualize()

if __name__ == "__main__":
    main()