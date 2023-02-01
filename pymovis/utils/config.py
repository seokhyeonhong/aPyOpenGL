import os
import json
from pymovis.vis.const import INCH_TO_METER

class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, filename):
        path = os.path.join(os.path.dirname(__file__), "..", "..", filename)
        with open(path, "r") as f:
            config = json.loads(f.read())
        
        return cls(config)

class DatasetConfig(Config):
    @classmethod
    def load(cls, filename):
        config = super().load(filename)

        # scale
        config.h_scale *= INCH_TO_METER
        config.v_scale *= INCH_TO_METER
        
        # filenames
        config.motion_pklname    = f"motion_length{config.window_length}_offset{config.window_offset}_fps{config.fps}.pkl"
        config.heightmap_pklname = f"heightmap_sparsity{config.sparsity}_mapsize{config.mapsize}.pkl"
        config.env_pklname       = f"env_length{config.window_length}_offset{config.window_offset}_fps{config.fps}_sparsity{config.sparsity}_mapsize{config.mapsize}_topk{config.top_k}.pkl"

        return config