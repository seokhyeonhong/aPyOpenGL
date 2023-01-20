import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import random

from pymovis.utils import util

class WindowDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.windows = []
        for f in sorted(os.listdir(path)):
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
        items = util.run_parallel_sync(self.__getitem__, indices)

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

""" Dataset pair of (X, Y) where X is a set of motion features and Y is a set of environment features """
class PairDataset(Dataset):
    def __init__(self, base_dir, train, X_name="motion", Y_name="envmap", window_size=50, window_offset=20, fps=30, sparsity=15, size=200, top_k_samples=10):
        self.base_dir = base_dir
        self.train = train
        self.X_name = X_name
        self.Y_name = Y_name

        self.window_size = window_size
        self.window_offset = window_offset
        self.fps = fps
        self.sparsity = sparsity
        self.size = size
        self.top_k_samples = top_k_samples

        self.X = np.load(os.path.join(base_dir, X_name, f"{'train' if train else 'test'}_size{window_size}_offset{window_offset}_fps{fps}.npy"))
        self.Y = np.load(os.path.join(base_dir, Y_name, f"{'train' if train else 'test'}_size{window_size}_offset{window_offset}_sparsity{sparsity}_size{size}_top{top_k_samples}.npy"))
        
        with open(os.path.join(base_dir, X_name, "skeleton.pkl"), "rb") as f:
            self.skeleton = pickle.load(f)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        rand = random.randint(0, self.top_k_samples - 1)
        return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(self.Y[idx, rand]).float(), rand
    
    def get_motion_statistics(self, dim):
        # load and return mean and std if they exist
        mean_path = os.path.join(self.base_dir, self.X_name, "mean.pt")
        std_path  = os.path.join(self.base_dir, self.X_name, "std.pt")

        if os.path.exists(mean_path) and os.path.exists(std_path):
            mean = torch.load(mean_path)
            std  = torch.load(std_path)
            return mean, std
        
        if not self.train:
            raise ValueError("Mean and std must be calculated and saved on training set first")

        # load all windows in parallel
        indices = list(range(len(self)))
        X       = [self.__getitem__(i)[0] for i in indices]
        X       = torch.stack(X, dim=0)
        
        # calculate mean and std
        mean = torch.mean(X, dim=dim)
        std  = torch.std(X, dim=dim) + 1e-8

        # save mean and std
        torch.save(mean, mean_path)
        torch.save(std,  std_path)

        return mean, std
    
    def motion_dim(self):
        return self.X.shape[-1]
    
    def env_dim(self):
        return self.Y.shape[-1]