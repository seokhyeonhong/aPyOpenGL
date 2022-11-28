import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

from pymovis.motion.data import bvh
from pymovis.motion.utils import npconst

class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        path,
        window_size,
        window_offset,
        align_at=None,
        v_forward=npconst.FORWARD(),
        v_up=npconst.UP(),
    ):
        self.path = path
        self.window_size = window_size
        self.window_offset = window_offset
        self.v_forward = np.array(v_forward, dtype=np.float32)
        self.v_up = np.array(v_up, dtype=np.float32)

        self.windows = self.load_windows(path)
        if align_at != None:
             for w in self.windows:
                w.align_by_frame(align_at, self.v_forward)
            
    def load_windows(self, path):
        # load pickle file if exists
        pkl_path = os.path.join(path, "dataset.pkl")
        if os.path.exists(pkl_path):
            print("Loading dataset from pickle file...")
            with open(pkl_path, "rb") as f:
                motions = pickle.load(f)
        else:
            files = []
            for f in os.listdir(path):
                if f.endswith(".bvh"):
                    files.append(os.path.join(path, f))
            motions = bvh.load_parallel(files, v_forward=self.v_forward, v_up=self.v_up)
            with open(pkl_path, "wb") as f:
                pickle.dump(motions, f)
            
        windows = []
        for idx, m in enumerate(motions):
            print(f"Creating windows... {idx+1}/{len(motions)}", end="\r")
            for start in range(0, m.num_frames - self.window_size, self.window_offset):
                end = start + self.window_size
                windows.append(m.make_window(start, end))
        print()
        return windows
    
    def get_skeleton(self):
        return self.windows[0].skeleton
    
    def set_dataset(self, *args):
        """
        Set the dataset to be used for training.
        Mandatory to use DataLoader.
        """
        self.dataset = np.concatenate(args, axis=-1)
        print(f"Dataset shape: {self.dataset.shape}")
    
    def get_local_R6(self):
        R6 = np.stack([w.get_local_R6() for w in self.windows], axis=0)
        return R6.reshape(*R6.shape[:-2], -1)

    def get_root_p(self):
        return np.stack([w.get_root_p() for w in self.windows], axis=0)

    def get_root_v(self):
        return np.stack([w.get_root_v() for w in self.windows], axis=0)

    def get_contacts(self, lfoot_idx, rfoot_idx):
        return np.stack([w.get_contacts(lfoot_idx, rfoot_idx) for w in self.windows], axis=0)

    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.dataset[idx])
    
    def sample_train_index(self, batch_size):
        return torch.randperm(len(self)).long().split(batch_size)