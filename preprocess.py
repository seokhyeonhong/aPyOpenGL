import os
import numpy as np
import pickle

from pymovis.motion.data import bvh
from pymovis.motion.ops import npmotion

# global variables
TRAIN = True
WINDOW_SIZE = 50
WINDOW_OFFSET = 20
LOAD_PATH = f"D:/data/LaFAN1/{'train' if TRAIN else 'test'}"
SAVE_PATH = f"data/{'train' if TRAIN else 'test'}_size{WINDOW_SIZE}_offset{WINDOW_OFFSET}"

def load_motions(path):
    files = []
    for f in os.listdir(path):
        if f.endswith(".bvh"):
            files.append(os.path.join(path, f))
    
    motions = bvh.load_parallel(files)
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

if __name__ == "__main__":
    motions = load_motions(LOAD_PATH)
    save_skeleton(motions[0], SAVE_PATH)
    save_windows(motions, SAVE_PATH)