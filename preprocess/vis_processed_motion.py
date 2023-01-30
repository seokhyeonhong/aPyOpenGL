import sys
sys.path.append(".")
sys.path.append("..")

import os
import pickle

from pymovis.motion.data.fbx import FBX
from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MotionApp
from pymovis.utils import util

""" Load from saved files """
def load_processed_data(save_dir, save_filename):
    # windows
    with open(os.path.join(save_dir, f"train_{save_filename}"), "rb") as f:
        train_windows = pickle.load(f)
    with open(os.path.join(save_dir, f"test_{save_filename}"), "rb") as f:
        test_windows = pickle.load(f)

    return train_windows["windows"], test_windows["windows"]

""" Main functions """
def main():
    # config
    motion_config, _ = util.config_parser()

    # load data
    train_windows, test_windows = load_processed_data(motion_config["save_dir"], motion_config["save_filename"])
    fbx = FBX("./data/models/model_skeleton.fbx")

    # visualize train data
    for window in train_windows:
        app_manager = AppManager()
        model = fbx.model()
        # print(window.name, window.type)
        app = MotionApp(window, model)
        app_manager.run(app)
    
    # visualize test data
    for window in test_windows:
        app_manager = AppManager()
        model = fbx.model()
        # print(window.name, window.type)
        app = MotionApp(window, model)
        app_manager.run(app)

if __name__ == "__main__":
    main()