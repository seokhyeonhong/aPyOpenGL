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
TRAIN = False
WINDOW_SIZE = 50
WINDOW_OFFSET = 20
LOAD_PATH = f"D:/data/LaFAN1/{'train' if TRAIN else 'test'}"
SAVE_PATH = f"data/{'train' if TRAIN else 'test'}_size{WINDOW_SIZE}_offset{WINDOW_OFFSET}"

def load_motions(path):
    files = []
    for f in os.listdir(path):
        if f.endswith(".bvh"):
            files.append(os.path.join(path, f))
    
    motions = bvh.load_parallel(files, v_forward=[0, 1, 0], v_up=[1, 0, 0])
    return motions

def save_skeleton(motion, path):
    skeleton = motion.skeleton
    filepath = os.path.join(path, "skeleton.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(skeleton, f)

def save_windows(motions, path):
    if not os.path.exists(path):
        os.makedirs(path)

    txt_count = 0
    for idx, m in enumerate(motions):
        print(f"Creating windows ... {idx+1}/{len(motions)}", end="\r")
        for start in range(0, m.num_frames - WINDOW_SIZE, WINDOW_OFFSET):
            end = start + WINDOW_SIZE
            window = m.make_window(start, end)
            window.align_by_frame(9)

            # ----------------------------------------------------------
            # Features to save. Modify this part to save other features.
            # Dimensions: (WINDOW_SIZE, D)
            # ----------------------------------------------------------
            local_R6 = npmotion.R6.from_R(window.local_R).reshape(WINDOW_SIZE, -1)
            root_p   = window.root_p.reshape(WINDOW_SIZE, -1)
            features = np.concatenate([local_R6, root_p], axis=-1)
            
            np.savetxt(os.path.join(path, f"{txt_count}.txt"), features, fmt="%.6f")
            txt_count += 1

class MotionApp(App):
    def __init__(self, motion):
        self.motion = motion

        """
        Animation parameters
        """
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
    motions = load_motions(LOAD_PATH)
    save_skeleton(motions[0], SAVE_PATH)
    save_windows(motions, SAVE_PATH)