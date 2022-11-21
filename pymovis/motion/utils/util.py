import torch
import numpy as np
import random

from functools import partial
from multiprocessing import Pool

def seed(x=777):
    torch.manual_seed(x)
    torch.cuda.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(x)
    random.seed(x)

def run_parallel(func, iterable, num_cpus=20, **kwargs):
    func_with_kwargs = partial(func, **kwargs)
    with Pool(processes=num_cpus) as pool:
        res = pool.map(func_with_kwargs, iterable)
    return res