import os
import pickle
import random

import torch
from torch.utils.data import Dataset


class MotionDataset(Dataset):
    def __init__(self, train, config):
        self.train  = train
        self.config = config

        with open(os.path.join(self.config.dataset_dir, f"{'train' if train else 'test'}_{self.config.motion_pklname}"), "rb") as f:
            self.data = pickle.load(f)["features"]
        with open(os.path.join(self.config.dataset_dir, "skeleton.pkl"), "rb") as f:
            self.skeleton = pickle.load(f)
        
        print("Motion dataset: ", self.data.shape)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float()
    
    def shape(self):
        return self.data.shape

    def statistics(self, dim):
        # load and return mean and std if they exist
        mean_path = os.path.join(self.config.dataset_dir, f"motion_mean_length{self.config.window_length}_offset{self.config.window_offset}_fps{self.config.fps}.pt")
        std_path  = os.path.join(self.config.dataset_dir, f"motion_std_length{self.config.window_length}_offset{self.config.window_offset}_fps{self.config.fps}.pt")

        if os.path.exists(mean_path) and os.path.exists(std_path):
            mean = torch.load(mean_path)
            std  = torch.load(std_path)
            return mean, std
        
        if not self.train:
            raise ValueError("Mean and std must be calculated and saved on training set first")

        # load motion features and calculate mean and std
        X    = torch.stack([self[i] for i in range(len(self))], dim=0)
        mean = torch.mean(X, dim=dim)
        std  = torch.std(X, dim=dim) + 1e-8

        # save mean and std
        torch.save(mean, mean_path)
        torch.save(std, std_path)

        return mean, std

class EnvmapDataset(Dataset):
    def __init__(self, train, config):
        self.train  = train
        self.config = config

        with open(os.path.join(self.config.dataset_dir, f"{'train' if train else 'test'}_{self.config.envmap_pklname}"), "rb") as f:
            self.data = pickle.load(f)
        
        # reshape to (B * top_K, T, D)
        self.data = self.data.reshape(-1, self.data.shape[2], self.data.shape[3])
        print("Envmap dataset: ", self.data.shape)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float()

    def shape(self):
        return self.data.shape

    def statistics(self, dim):
        # load and return mean and std if they exist
        mean_path = os.path.join(self.config.dataset_dir, f"envmap_mean_{os.path.splitext(self.config.envmap_pklname)[0]}.pt")
        std_path  = os.path.join(self.config.dataset_dir, f"envmap_std_{os.path.splitext(self.config.envmap_pklname)[0]}.pt")

        if os.path.exists(mean_path) and os.path.exists(std_path):
            mean = torch.load(mean_path)
            std  = torch.load(std_path)
            return mean, std
        
        if not self.train:
            raise ValueError("Mean and std must be calculated and saved on training set first")

        # load envmap features and calculate mean and std
        X    = torch.stack([self[i] for i in range(len(self))], dim=0)
        mean = torch.mean(X, dim=dim)
        std  = torch.std(X, dim=dim) + 1e-8

        # save mean and std
        torch.save(mean, mean_path)
        torch.save(std, std_path)

        return mean, std

""" Paired dataset for motion and envmap features """
class PairDataset(Dataset):
    def __init__(self, train, config):
        self.train  = train
        self.config = config

        self.motion_dset   = MotionDataset(train, config)
        self.envmap_dset   = EnvmapDataset(train, config)

        if self.motion_dset.shape()[0] * self.config.top_k != self.envmap_dset.shape()[0]:
            raise ValueError(f"Motion and envmap datasets must have the same data samples, but got {self.motion_dset.shape()[0]} and {self.envmap_dset.shape()[0]}")
        if self.motion_dset.shape()[1] != self.envmap_dset.shape()[1]:
            raise ValueError(f"Motion and envmap datasets must have the same length, but got {self.motion_dset.shape()[1:]} and {self.envmap_dset.shape()[1:]}")
        
    def __len__(self):
        return len(self.motion_dset)
    
    def __getitem__(self, idx):
        rand = random.randint(0, self.config.top_k - 1)
        x = self.motion_dset[idx]
        y = self.envmap_dset[idx * self.config.top_k + rand]
        return x, y, rand
    
    def motion_statistics(self, dim):
        return self.motion_dset.statistics(dim)
    
    def envmap_statistics(self, dim):
        return self.envmap_dset.statistics(dim)

    def motion_dim(self):
        return self.motion_dset.shape()[-1]
    
    def env_dim(self):
        return self.envmap_dset.shape()[-1]