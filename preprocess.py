import os
import numpy as np
import pickle

from pymovis.motion.data import bvh
from pymovis.motion.ops import npmotion

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import App
from pymovis.vis.render import Render
import glfw, glm

# global variables
WINDOW_SIZE = 50
WINDOW_OFFSET = 20
DATASET_PATH = f"data/PFNN"

def load_motions(subset):
    if subset not in ["train", "test"]:
        raise ValueError("subset must be either 'train' or 'test'")

    files = []
    load_dir = os.path.join(DATASET_PATH, subset)
    for f in os.listdir(load_dir):
        if f.endswith(".bvh"):
            files.append(os.path.join(load_dir, f))
    
    motions = bvh.load_parallel(files, v_forward=[0, 1, 0], v_up=[1, 0, 0])
    return motions

def save_skeleton(skeleton):
    save_path = os.path.join(DATASET_PATH, "skeleton.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(skeleton, f)

def save_windows(motions, subset):
    if subset not in ["train", "test"]:
        raise ValueError("subset must be either 'train' or 'test'")

    save_dir = os.path.join(DATASET_PATH, f"{subset}_txt")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    txt_count = 0
    print()
    for idx, m in enumerate(motions):
        print(f"Saving pairs ... {idx+1}/{len(motions)}", end="\r")
        for start in range(0, m.num_frames - WINDOW_SIZE, WINDOW_OFFSET):
            end = start + WINDOW_SIZE
            window = m.make_window(start, end)
            window.align_by_frame(9)

            # -------------- Modify this part to save other features --------------
            # Features to save. Modify this part to save other features.
            # Dimensions: (WINDOW_SIZE, D)
            # ---------------------------------------------------------------------
            local_R6   = npmotion.R6.from_R(window.local_R).reshape(WINDOW_SIZE, -1)
            root_p     = window.root_p.reshape(WINDOW_SIZE, -1)
            base       = np.stack([pose.base for pose in window.poses], axis=0)
            forward    = np.stack([pose.forward for pose in window.poses], axis=0)
            lr_heights = np.zeros_like(forward[:, :2]) if "walk" in m.name else np.ones_like(forward[:, :2])
            
            geometry   = np.concatenate([base, forward, lr_heights], axis=-1)
            motion     = np.concatenate([local_R6, root_p], axis=-1)
            
            if not os.path.exists(os.path.join(save_dir, "geometry")):
                os.makedirs(os.path.join(save_dir, "geometry"))
            np.savetxt(os.path.join(save_dir, "geometry", f"{txt_count}.txt"), geometry)

            if not os.path.exists(os.path.join(save_dir, "motion")):
                os.makedirs(os.path.join(save_dir, "motion"))
            np.savetxt(os.path.join(save_dir, "motion", f"{txt_count}.txt"), motion)

            txt_count += 1

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
    train_motions = load_motions("train")
    test_motions = load_motions("test")
    
    skeleton = train_motions[0].skeleton
    save_skeleton(skeleton)

    save_windows(train_motions, "train")
    save_windows(test_motions, "test")