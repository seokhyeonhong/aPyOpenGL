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

class PairDataset(Dataset):
    """ Dataset pair of (X, Y) where X is a motion and Y is a environment feature """
    def __init__(self, dataset_dir, train):
        self.dataset_dir = dataset_dir
        self.train = train

        self.path = os.path.join(dataset_dir, "train" if train else "test", "features")
        self.data = []
        for f in os.listdir(self.path):
            self.data.append(os.path.join(self.path, f))
        
        with open(os.path.join(dataset_dir, "skeleton.pkl"), "rb") as f:
            self.skeleton = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        with open(self.data[idx], "rb") as f:
            X, Y = pickle.load(f)
            X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
            return X, Y
    
    def get_motion_statistics(self, dim):
        # load and return mean and std if they exist
        X_mean_path = os.path.join(self.dataset_dir, "X_mean.pt")
        X_std_path  = os.path.join(self.dataset_dir, "X_std.pt")
        Y_mean_path = os.path.join(self.dataset_dir, "Y_mean.pt")
        Y_std_path  = os.path.join(self.dataset_dir, "Y_std.pt")

        if os.path.exists(X_mean_path) and os.path.exists(X_std_path) and os.path.exists(Y_mean_path) and os.path.exists(Y_std_path):
            X_mean = torch.load(X_mean_path)
            X_std  = torch.load(X_std_path)
            Y_mean = torch.load(Y_mean_path)
            Y_std  = torch.load(Y_std_path)
            return X_mean, X_std, Y_mean, Y_std
        
        if not self.train:
            raise ValueError("Mean and std must be calculated and saved on training set first")

        # load all windows in parallel
        indices = list(range(len(self)))
        items   = util.run_parallel_async(self.__getitem__, indices)
        X, Y    = zip(*items)
        
        X       = torch.stack(X, dim=0)
        Y       = torch.stack(Y, dim=0)
        
        # calculate mean and std
        X_mean = torch.mean(X, dim=dim)
        X_std  = torch.std(X, dim=dim) + 1e-8
        Y_mean = torch.mean(Y, dim=dim)
        Y_std  = torch.std(Y, dim=dim) + 1e-8

        # save mean and std
        torch.save(X_mean, X_mean_path)
        torch.save(X_std,  X_std_path)
        torch.save(Y_mean, Y_mean_path)
        torch.save(Y_std,  Y_std_path)

        return X_mean, X_std, Y_mean, Y_std