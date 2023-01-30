import os
import torch
import numpy as np
import random
from tqdm import tqdm

from functools import partial
import multiprocessing as mp

def seed(x=777):
    torch.manual_seed(x)
    torch.cuda.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(x)
    random.seed(x)

def run_parallel_sync(func, iterable, num_cpus=mp.cpu_count(), desc=None, **kwargs):
    if desc is not None:
        print(f"{desc} in sync [CPU: {num_cpus}]")

    func_with_kwargs = partial(func, **kwargs)
    with mp.Pool(num_cpus) as pool:
        res = pool.map(func_with_kwargs, iterable) if iterable is not None else pool.map(func_with_kwargs)

    return res

def run_parallel_async(func, iterable, num_cpus=mp.cpu_count(), desc=None, **kwargs):
    if desc is not None:
        print(f"{desc} in async [CPU: {num_cpus}]")

    func_with_kwargs = partial(func, **kwargs)
    with mp.Pool(num_cpus) as pool:
        res = list(tqdm(pool.imap(func_with_kwargs, iterable) if iterable is not None else pool.imap(func_with_kwargs), total=len(iterable)))

    return res

def clamp(x, min_val, max_val):
    return max(min_val, min(x, max_val))

def config_parser():
    import configparser
    config = configparser.ConfigParser()
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs")
    config.read(os.path.join(config_dir, "config.ini"))

    # config parser for motion data
    window_length = config.getint("Motion", "window_length")
    window_offset = config.getint("Motion", "window_offset")
    fps = config.getint("Motion", "fps")
    motion_load_dir = config.get("Motion", "load_dir")
    motion_save_dir = config.get("Motion", "save_dir")
    motion_save_filename = f"length{window_length}_offset{window_offset}_fps{fps}.pkl"

    # config parser for heightmap data
    sparsity = config.getint("Heightmap", "sparsity")
    mapsize = config.getint("Heightmap", "mapsize")
    top_k = config.getint("Heightmap", "top_k")
    h_scale = config.getfloat("Heightmap", "h_scale")
    v_scale = config.getfloat("Heightmap", "v_scale")
    heightmap_load_dir = config.get("Heightmap", "load_dir")
    heightmap_save_dir = config.get("Heightmap", "save_dir")
    heightmap_save_filename = f"sparsity{sparsity}_mapsize{mapsize}.pkl"

    motion_config = {
        "window_length": window_length,
        "window_offset": window_offset,
        "fps": fps,
        "load_dir": motion_load_dir,
        "save_dir": motion_save_dir,
        "save_filename": motion_save_filename
    }

    heightmap_config = {
        "sparsity": sparsity,
        "mapsize": mapsize,
        "top_k": top_k,
        "h_scale": h_scale,
        "v_scale": v_scale,
        "load_dir": heightmap_load_dir,
        "save_dir": heightmap_save_dir,
        "save_filename": heightmap_save_filename
    }
    
    return motion_config, heightmap_config