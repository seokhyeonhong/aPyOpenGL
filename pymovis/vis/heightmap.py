import numpy as np
import torch

from pymovis.vis.core import VertexGL, VAO
from pymovis.vis.const import INCH_TO_METER

def _sample_height_numpy(heightmap, x, z, h_scale, v_scale):
    h, w = heightmap.shape[-2:]

    x_ = (x / h_scale) + (w / 2)
    z_ = (z / h_scale) + (h / 2)

    a0 = np.fmod(x_, 1)
    a1 = np.fmod(z_, 1)

    x0, x1 = np.floor(x_).astype(np.int32), np.ceil(x_).astype(np.int32)
    z0, z1 = np.floor(z_).astype(np.int32), np.ceil(z_).astype(np.int32)

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    z0 = np.clip(z0, 0, h - 1)
    z1 = np.clip(z1, 0, h - 1)

    s0 = heightmap[..., z0, x0]
    s1 = heightmap[..., z0, x1]
    s2 = heightmap[..., z1, x0]
    s3 = heightmap[..., z1, x1]

    return v_scale * ((s0 * (1 - a0) + s1 * a0) * (1 - a1) + (s2 * (1 - a0) + s3 * a0) * a1)

def _sample_height_torch(heightmap, x, z, h_scale, v_scale):
    h, w = heightmap.shape[-2:]

    x_ = (x / h_scale) + (w / 2)
    z_ = (z / h_scale) + (h / 2)

    a0 = torch.fmod(x_, 1)
    a1 = torch.fmod(z_, 1)

    x0, x1 = torch.floor(x_).long(), torch.ceil(x_).long()
    z0, z1 = torch.floor(z_).long(), torch.ceil(z_).long()

    x0 = torch.clamp(x0, 0, w - 1)
    x1 = torch.clamp(x1, 0, w - 1)
    z0 = torch.clamp(z0, 0, h - 1)
    z1 = torch.clamp(z1, 0, h - 1)

    s0 = heightmap[..., z0, x0]
    s1 = heightmap[..., z0, x1]
    s2 = heightmap[..., z1, x0]
    s3 = heightmap[..., z1, x1]

    return v_scale * ((s0 * (1 - a0) + s1 * a0) * (1 - a1) + (s2 * (1 - a0) + s3 * a0) * a1)

class Heightmap:
    def __init__(self, data, h_scale=INCH_TO_METER, v_scale=INCH_TO_METER, offset=None):
        self.data = data
        self.h_scale = h_scale
        self.v_scale = v_scale
        self.offset = offset
        self.__init_mesh()
    
    @classmethod
    def load_from_file(cls, filename, h_scale=INCH_TO_METER, v_scale=INCH_TO_METER, offset=None):
        data = np.loadtxt(filename, dtype=np.float32)
        return cls(data, h_scale, v_scale, offset)

    def __init_mesh(self):
        # create vertices from the heightmap data
        h, w = self.data.shape

        self.offset = np.sum(self.data) / (h * w) if self.offset is None else self.offset
        self.data -= self.offset
        print(f"Loaded Heightmap: {h}x{w} points ({self.h_scale * h:.4f}m x {self.h_scale * w:.4f}m)")

        # vertex positions
        px = self.h_scale * (np.arange(w, dtype=np.float32) - w / 2)
        pz = self.h_scale * (np.arange(h, dtype=np.float32) - h / 2)
        px, pz = np.meshgrid(px, pz)

        py = self.sample_height(self.data, px, pz, self.h_scale, self.v_scale)
        positions = np.stack([px, py, pz], axis=-1)
        
        # vertex normals
        normals = np.empty((h, w, 3), dtype=np.float32)

        cross1 = np.cross(positions[2:, 1:-1] - positions[1:-1, 1:-1], positions[1:-1, 2:] - positions[1:-1, 1:-1])
        cross2 = np.cross(positions[:-2, 1:-1] - positions[1:-1, 1:-1], positions[1:-1, :-2] - positions[1:-1, 1:-1])
        cross = (cross1 + cross2) * 0.5
        cross = cross / (np.linalg.norm(cross, axis=-1, keepdims=True) + 1e-8)

        normals[1:-1, 1:-1] = cross
        normals[0, :] = normals[-1, :] = np.array([0, 1, 0], dtype=np.float32)
        normals[:, 0] = normals[:, -1] = np.array([0, 1, 0], dtype=np.float32)
        
        # vertex UV coordinates
        uvs = np.stack([px, pz], axis=-1)

        # vertex indices
        indices = np.empty((h - 1, w - 1, 6), dtype=np.int32)
        indices[..., 0] = np.arange(h * w).reshape(h, w)[:-1, :-1]
        indices[..., 1] = indices[..., 4] = indices[..., 0] + w
        indices[..., 2] = indices[..., 3] = indices[..., 0] + 1
        indices[..., 5] = indices[..., 0] + w + 1
        indices = indices.flatten()

        # vertices and vao
        vertices = VertexGL.make_vertex_array(positions.reshape(-1, 3), normals.reshape(-1, 3), uvs.reshape(-1, 2))
        self.vao = VAO.from_vertex_array(vertices, indices)

    @staticmethod
    def sample_height(heightmap, x, z, h_scale, v_scale):
        if isinstance(heightmap, np.ndarray):
            return _sample_height_numpy(heightmap, x, z, h_scale, v_scale)
        elif isinstance(heightmap, torch.Tensor):
            return _sample_height_torch(heightmap, x, z, h_scale, v_scale)
        else:
            raise TypeError(f"Unsupported type: {type(heightmap)}")