import sys
sys.path.append(".")

import os
import pickle

import numpy as np

from pymovis.motion.ops import npmotion
from pymovis.motion.core import Motion

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MotionApp

""" Global variables for the dataset """
WINDOW_SIZE   = 50
WINDOW_OFFSET = 20
FPS           = 30
MOTION_DIR    = "./data/animations"
SAVE_DIR      = "./data/dataset/motion"

""" Load from saved files """
def load_processed_data():
    # skeleton
    with open(os.path.join(SAVE_DIR, "skeleton.pkl"), "rb") as f:
        skeleton = pickle.load(f)

    # windows
    train_windows = np.load(os.path.join(SAVE_DIR, f"train_size{WINDOW_SIZE}_offset{WINDOW_OFFSET}_fps{FPS}.npy"))
    test_windows  = np.load(os.path.join(SAVE_DIR, f"test_size{WINDOW_SIZE}_offset{WINDOW_OFFSET}_fps{FPS}.npy"))

    return skeleton, train_windows, test_windows

""" Main functions """
def visualize(train=True, test=True):
    skeleton, train_windows, test_windows = load_processed_data()

    if train:
        for window in train_windows:
            local_R6, root_p = window[:, :-3], window[:, -3:]
            local_R = npmotion.R6_to_R(local_R6.reshape(-1, 6)).reshape(WINDOW_SIZE, -1, 3, 3)
            motion = Motion.from_numpy(skeleton, local_R, root_p, fps=FPS)

            app_manager = AppManager()
            app = MotionApp(motion)
            app_manager.run(app)
    
    if test:
        for window in test_windows:
            local_R6, root_p = window[:, :-3], window[:, -3:]
            local_R = npmotion.R6_to_R(local_R6.reshape(-1, 6)).reshape(WINDOW_SIZE, -1, 3, 3)
            motion = Motion.from_numpy(skeleton, local_R, root_p, fps=FPS)

            app_manager = AppManager()
            app = MotionApp(motion)
            app_manager.run(app)

def main():
    visualize()

if __name__ == "__main__":
    main()