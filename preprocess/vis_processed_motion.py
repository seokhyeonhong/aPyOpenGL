import sys
sys.path.append(".")
sys.path.append("..")

import os
import pickle

from pymovis.motion.data.fbx import FBX

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MotionApp

""" Global variables for the dataset """
WINDOW_SIZE   = 50
WINDOW_OFFSET = 20
FPS           = 30
SAVE_DIR      = "./data/dataset/motion"
SAVE_FILENAME = f"length{WINDOW_SIZE}_offset{WINDOW_OFFSET}_fps{FPS}.pkl"

""" Load from saved files """
def load_processed_data():
    # windows
    with open(os.path.join(SAVE_DIR, f"train_{SAVE_FILENAME}"), "rb") as f:
        train_windows = pickle.load(f)
    with open(os.path.join(SAVE_DIR, f"test_{SAVE_FILENAME}"), "rb") as f:
        test_windows = pickle.load(f)

    return train_windows["windows"], test_windows["windows"]

""" Main functions """
def visualize(train=True, test=True):
    train_windows, test_windows = load_processed_data()
    fbx = FBX("./data/models/model_skeleton.fbx")

    if train:
        for window in train_windows:
            app_manager = AppManager()
            model = fbx.model()
            print(window.name, window.type)
            app = MotionApp(window, model)
            app_manager.run(app)
    
    if test:
        for window in test_windows:
            app_manager = AppManager()
            model = fbx.model()
            # print(window.name, window.type)
            app = MotionApp(window, model)
            app_manager.run(app)

def main():
    visualize()

if __name__ == "__main__":
    main()