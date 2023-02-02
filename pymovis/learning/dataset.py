import os
import pickle
import torch
from torch.utils.data import Dataset

from ops import mathops

class MotionDataset(Dataset):
    def __init__(self, train, config):
        self.train  = train
        self.config = config
        
        split = "train" if train else "test"
        with open(os.path.join(self.config.dataset_dir, f"{split}_{self.config.motion_pklname}"), "rb") as f:
            self.data = pickle.load(f)["features"]
        with open(os.path.join(self.config.dataset_dir, "skeleton.pkl"), "rb") as f:
            self.skeleton = pickle.load(f)
        
        print("Motion dataset: ", self.data.shape)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float()
    
    @property
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

class EnvironmentDataset(Dataset):
    def __init__(self, train, config):
        self.train  = train
        self.config = config

        split = "train" if train else "test"
        with open(os.path.join(self.config.dataset_dir, f"{split}_{self.config.env_pklname}"), "rb") as f:
            data = pickle.load(f)
        
        # N: number of samples, T: number of frames, D: dimension of feature
        self.patch     = data["patches"] # (N * top_K, mapsize, mapsize)
        self.env_state = data["env"]     # (N * top_K, T, D)

        if len(self.patch) != len(self.env_state):
            raise ValueError(f"Number of patches ({len(self.patch)}) and number of environment states ({len(self.env_state)}) do not match")
        print("Patch dataset: ", self.patch.shape)
        print("Environment state dataset: ", self.env_state.shape)
    
    def __len__(self):
        return len(self.patch)

    def __getitem__(self, idx):
        patch = torch.from_numpy(self.patch[idx]).float()
        env_state = torch.from_numpy(self.env_state[idx]).float()
        return patch, env_state

    @property
    def patch_shape(self):
        return self.patch.shape
    
    @property
    def env_shape(self):
        return self.env_state.shape

    def env_statistics(self, dim):
        # load and return mean and std if they exist
        mean_path = os.path.join(self.config.dataset_dir, f"env_mean_{os.path.splitext(self.config.env_pklname)[0]}.pt")
        std_path  = os.path.join(self.config.dataset_dir, f"env_std_{os.path.splitext(self.config.env_pklname)[0]}.pt")

        if os.path.exists(mean_path) and os.path.exists(std_path):
            mean = torch.load(mean_path)
            std  = torch.load(std_path)
            return mean, std
        
        if not self.train:
            raise ValueError("Mean and std must be calculated and saved on training set first")

        # load envmap features and calculate mean and std
        envs = [self[i][1] for i in range(len(self))]
        envs = torch.stack(envs, dim=0)
        mean = torch.mean(envs, dim=dim)
        std  = torch.std(envs, dim=dim) + 1e-8

        # save mean and std
        torch.save(mean, mean_path)
        torch.save(std, std_path)

        return mean, std
    
    def disp_statistics(self, dim):
        # load and return mean and std if they exist
        mean_path = os.path.join(self.config.dataset_dir, f"disp_mean_{os.path.splitext(self.config.env_pklname)[0]}.pt")
        std_path  = os.path.join(self.config.dataset_dir, f"disp_std_{os.path.splitext(self.config.env_pklname)[0]}.pt")

        if os.path.exists(mean_path) and os.path.exists(std_path):
            mean = torch.load(mean_path)
            std  = torch.load(std_path)
            return mean, std
        
        if not self.train:
            raise ValueError("Mean and std must be calculated and saved on training set first")

        # load displacement features
        envs = [self[i][1] for i in range(len(self))]
        envs = torch.stack(envs, dim=0)
        
        disp_xz = envs[:, 1:, (0, 2)] - envs[:, :-1, (0, 2)]
        disp_forward = mathops.signed_angle(envs[:, :-1, 3:6], envs[:, 1:, 3:6], dim=-1).unsqueeze(-1)
        disp = torch.cat([disp_xz, disp_forward], dim=-1)

        mean = torch.mean(disp, dim=dim)
        std  = torch.std(disp, dim=dim) + 1e-8

        # save mean and std
        torch.save(mean, mean_path)
        torch.save(std, std_path)

        return mean, std

""" Paired dataset for motion and envmap features """
class PairDataset(Dataset):
    def __init__(self, train, config):
        self.train  = train
        self.config = config

        self.motion_dset = MotionDataset(train, config)
        self.env_dset    = EnvironmentDataset(train, config)
        self.skeleton    = self.motion_dset.skeleton

        if self.motion_dset.shape[0] * self.config.top_k != self.env_dset.env_shape[0]:
            raise ValueError(f"Motion and environment datasets must have the same data samples, but got {self.motion_dset.shape[0]} and {self.env_dset.env_shape[0]}")
        if self.motion_dset.shape[1] != self.env_dset.env_shape[1]:
            raise ValueError(f"Motion and environment datasets must have the same length, but got {self.motion_dset.shape[1]} and {self.env_dset.env_shape[1]}")
        
    def __len__(self):
        return len(self.motion_dset)
    
    def __getitem__(self, idx):
        rand = torch.randint(0, self.config.top_k, (1,)).item()
        motion = self.motion_dset[idx]
        patch, env = self.env_dset[idx * self.config.top_k + rand]
        return motion, patch, env, rand

    def motion_statistics(self, dim):
        return self.motion_dset.statistics(dim)
    
    def env_statistics(self, dim):
        return self.env_dset.env_statistics(dim)
    
    def disp_statistics(self, dim):
        return self.env_dset.disp_statistics(dim)

    @property
    def motion_shape(self):
        return self.motion_dset.shape

    @property
    def env_shape(self):
        return self.env_dset.env_shape

    @property
    def patch_shape(self):
        return self.env_dset.patch_shape