import os
import shutil
import pickle

import numpy as np

from pymovis.motion.data import bvh
from pymovis.motion.ops import npmotion
from pymovis.motion.core import Motion

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import App
from pymovis.vis.render import Render
import glfw, glm

"""
TODO
1. save mean and std
2. rename the directory as {train | test}_size{WINDOW_SIZE}_offset{WINDOW_OFFSET}_fps{FPS}
"""

# dataset parameters
WINDOW_SIZE = 120
WINDOW_OFFSET = 20
FPS = 30
DATASET_PATH = f"data/NSM"

def load_motions(split):
    if split not in ["train", "test"]:
        raise ValueError("split must be either 'train' or 'test'")

    files = []
    load_dir = os.path.join(DATASET_PATH, split)
    for f in os.listdir(load_dir):
        if f.endswith(".bvh"):
            files.append(os.path.join(load_dir, f))
    
    motions = bvh.load_parallel(files, v_forward=[0, -1, 0], v_up=[1, 0, 0])
    return motions

def save_skeleton(skeleton):
    save_path = os.path.join(DATASET_PATH, "skeleton.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(skeleton, f)

def save_windows(motions, split):
    if split not in ["train", "test"]:
        raise ValueError("split must be either 'train' or 'test'")

    save_dir = os.path.join(DATASET_PATH, f"{split}_size{WINDOW_SIZE}_offset{WINDOW_OFFSET}_fps{FPS}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    geometry_list = []
    motion_list = []
    for idx, m in enumerate(motions):
        print(f"Making {split} list ... {idx+1} / {len(motions)}", end="\r")
        for start in range(0, m.num_frames - WINDOW_SIZE, WINDOW_OFFSET):
            end = start + WINDOW_SIZE
            window = m.make_window(start, end)

            # -------------- Modify this part to save other features --------------
            # Features to save. Modify this part to save other features.
            # Dimensions: (WINDOW_SIZE, D)
            window.align_by_frame(9)
            
            local_R6   = npmotion.R6.from_R(window.local_R).reshape(WINDOW_SIZE, -1)
            root_p     = window.root_p.reshape(WINDOW_SIZE, -1)
            base       = np.stack([pose.base for pose in window.poses], axis=0)
            forward    = np.stack([pose.forward for pose in window.poses], axis=0)
            lr_heights = np.zeros_like(forward[:, :2]) if "Armchair" in m.name else np.ones_like(forward[:, :2])
            
            geometry   = np.concatenate([base, forward, lr_heights], axis=-1)
            motion     = np.concatenate([local_R6, root_p], axis=-1)
            
            geometry_list.append(geometry)
            motion_list.append(motion)
            # if not os.path.exists(os.path.join(save_dir, "geometry")):
            #     os.makedirs(os.path.join(save_dir, "geometry"))
            # np.savetxt(os.path.join(save_dir, "geometry", f"{txt_count}.txt"), geometry)

            # if not os.path.exists(os.path.join(save_dir, "motion")):
            #     os.makedirs(os.path.join(save_dir, "motion"))
            # np.savetxt(os.path.join(save_dir, "motion", f"{txt_count}.txt"), motion)
            # ---------------------------------------------------------------------

            # txt_count += 1

            # app_manager = AppManager.initialize()
            # app = MotionApp(window)
            # app_manager.run(app)
    
    geometry_list = np.stack(geometry_list, axis=0)
    motion_list = np.stack(motion_list, axis=0)
    np.save(os.path.join(save_dir, "geometry.npy"), geometry_list)
    np.save(os.path.join(save_dir, "motion.npy"), motion_list)
    print()

def main():
    # load motions
    train_motions = load_motions("train")
    test_motions = load_motions("test")
    
    # save skeleton
    skeleton = train_motions[0].skeleton
    save_skeleton(skeleton)

    # save windows
    save_windows(train_motions, "train")
    save_windows(test_motions, "test")

    # save mean and std

class MotionApp(App):
    def __init__(self, motion):
        self.motion = motion

        self.frame = 0
        self.play = True
        super().__init__()
    
    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_LEFT:
            self.frame = max(self.frame - 1, 0)
            self.play = False
        elif key == glfw.KEY_RIGHT:
            self.frame = min(self.frame + 1, self.motion.num_frames - 1)
            self.play = False
        elif key == glfw.KEY_Z and action == glfw.PRESS:
            self.frame = max(self.frame - 1, 0)
            self.play = False
        elif key == glfw.KEY_X and action == glfw.PRESS:
            self.frame = min(self.frame + 1, self.motion.num_frames - 1)
            self.play = False
        elif key == glfw.KEY_P and action == glfw.PRESS:
            glfw.set_time(0)
            self.play = True

        super().key_callback(window, key, scancode, action, mods)

    def render(self):
        Render.plane().set_scale(50).set_uv_repeat(5).set_texture("example.png").draw()
        Render.arrow().draw()
        Render.text(self.frame).draw()
        if self.play:
            self.frame = min(int(glfw.get_time() * 30), self.motion.num_frames - 1)
            self.motion.render_by_frame(self.frame)
        else:
            self.motion.render_by_frame(self.frame)

if __name__ == "__main__":
    main()