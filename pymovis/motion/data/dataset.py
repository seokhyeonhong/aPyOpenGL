import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

from pymovis.motion.core import Skeleton
from pymovis.utils import torchconst, util

class WindowDataset(Dataset):
    def __init__(self, path):
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

class PairDataset(Dataset):
    def __init__(self, dataset_path, target_x, target_y, train=True, window_size=50, window_offset=20, fps=30):
        self.dataset_path = dataset_path
        self.target_x = target_x
        self.target_y = target_y
        self.train = train

        # load input and output file paths
        base_path = os.path.join(dataset_path, f"{'train' if train else 'test'}_size{window_size}_offset{window_offset}_fps{fps}")
        self.X = torch.from_numpy(np.load(os.path.join(base_path, f"{target_x}.npy"))).float()
        self.Y = torch.from_numpy(np.load(os.path.join(base_path, f"{target_y}.npy"))).float()

        if len(self.X) != len(self.Y):
            raise ValueError("Number of input and output files must be equal")

        # load skeleton
        with open(os.path.join(dataset_path, "skeleton.pkl"), "rb") as f:
            self.skeleton = pickle.load(f)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def get_mean_std(self, dim, target):
        if target not in [self.target_x, self.target_y]:
            raise ValueError(f"target must be either '{self.target_x}' or '{self.target_y}'")
        
        # load and return mean and std if they exist
        mean_path = os.path.join(self.dataset_path, f"{target}_mean.pt")
        std_path = os.path.join(self.dataset_path, f"{target}_std.pt")
        if os.path.exists(mean_path) and os.path.exists(std_path):
            mean = torch.load(mean_path)
            std = torch.load(std_path)
            return mean, std
        
        # raise error if it is not a training set
        if not self.train:
            raise ValueError("Mean and std must be calculated and saved on training set first")

        # load all windows in parallel
        indices = list(range(len(self)))
        items = util.run_parallel(self.__getitem__, indices)
        Xs = [item[0] for item in items]
        Ys = [item[1] for item in items]

        # calculate mean and std
        input = torch.stack(Xs if target == self.target_x else Ys, dim=0)
        mean = torch.mean(input, dim=dim)
        std = torch.std(input, dim=dim) + 1e-8

        # save mean and std
        torch.save(mean, mean_path)
        torch.save(std, std_path)

        return mean, std
    
    def get_feature_dim(self, target):
        if target not in [self.target_x, self.target_y]:
            raise ValueError(f"target must be either '{self.target_x}' or '{self.target_y}'")
        X, Y = self[0]
        return X.shape[-1] if target == self.target_x else Y.shape[-1]