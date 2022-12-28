import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

from pymovis.motion.core import Skeleton
from pymovis.utils import torchconst, util

class WindowDataset(Dataset):
    def __init__(
        self,
        path
    ):
        self.path = path
        self.windows = []
        for f in os.listdir(path):
            if f.endswith(".txt"):
                self.windows.append(os.path.join(path, f))
        
        with open(os.path.join(path, "skeleton.pkl"), "rb") as f:
            self.skeleton = pickle.load(f)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        res = np.loadtxt(self.windows[idx], dtype=np.float32)
        return torch.from_numpy(res)
    
    def get_mean_std(self, dim):
        # load and return mean and std if they exist
        mean_path = os.path.join(self.path, "mean.pt")
        std_path = os.path.join(self.path, "std.pt")
        if os.path.exists(mean_path) and os.path.exists(std_path):
            mean = torch.load(mean_path)
            std = torch.load(std_path)
            return mean, std

        # load all windows in parallel
        indices = list(range(len(self)))
        items = util.run_parallel(self.__getitem__, indices)

        # calculate mean and std
        items = torch.stack(items, dim=0)
        mean = torch.mean(items, dim=dim)
        std = torch.std(items, dim=dim) + 1e-6

        # save mean and std
        torch.save(mean, mean_path)
        torch.save(std, std_path)

        return mean, std

    def get_feature_dim(self):
        return self[0].shape[-1]