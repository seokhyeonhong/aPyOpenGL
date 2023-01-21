import numpy as np

from pymovis.vis.core import Vertex, VAO, Mesh
from pymovis.vis.render import Render
from pymovis.vis.glconst import INCH_TO_METER

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
        print(f"Loaded Heightmap: {h}x{w} points ({self.h_scale * h:.4f}m x {self.h_scale * w:.4f}m)")

        vertices = [Vertex() for _ in range(h * w)]

        # vertex positions
        px = self.h_scale * (np.arange(w) - w / 2)
        pz = self.h_scale * (np.arange(h) - h / 2)
        px, pz = np.meshgrid(px, pz)

        py = self.sample_height(px, pz)
        positions = np.stack([px, py, pz], axis=-1)

        for i, pos in enumerate(positions.reshape(-1, 3)):
            vertices[i].position = pos
        
        # vertex normals
        normals = np.empty((h, w, 3), dtype=np.float32)

        cross1 = np.cross(positions[2:, 1:-1] - positions[1:-1, 1:-1], positions[1:-1, 2:] - positions[1:-1, 1:-1])
        cross2 = np.cross(positions[:-2, 1:-1] - positions[1:-1, 1:-1], positions[1:-1, :-2] - positions[1:-1, 1:-1])
        cross = (cross1 + cross2) * 0.5
        cross = cross / (np.linalg.norm(cross, axis=-1, keepdims=True) + 1e-8)

        normals[1:-1, 1:-1] = cross
        normals[0, :] = normals[-1, :] = np.array([0, 1, 0])
        normals[:, 0] = normals[:, -1] = np.array([0, 1, 0])

        for i, normal in enumerate(normals.reshape(-1, 3)):
            vertices[i].normal = normal
        
        # vertex UV coordinates
        uvs = np.stack([px, pz], axis=-1)

        for i, uv in enumerate(uvs.reshape(-1, 2)):
            vertices[i].uv= uv

        # vertex indices
        indices = np.empty((h - 1, w - 1, 6), dtype=np.int32)
        indices[..., 0] = np.arange(h * w).reshape(h, w)[:-1, :-1]
        indices[..., 1] = indices[..., 4] = indices[..., 0] + w
        indices[..., 2] = indices[..., 3] = indices[..., 0] + 1
        indices[..., 5] = indices[..., 0] + w + 1
        indices = indices.flatten()

        # VAO and mesh
        vao = VAO.from_vertex_array(vertices, indices)
        self.mesh = Mesh(vao, vertices, indices)

    def sample_height(self, x, z):
        h, w  = self.data.shape

        x = (x / self.h_scale) + (w / 2)
        z = (z / self.h_scale) + (h / 2)

        a0 = np.fmod(x, 1)
        a1 = np.fmod(z, 1)

        x0, x1 = np.floor(x).astype(np.int32), np.ceil(x).astype(np.int32)
        z0, z1 = np.floor(z).astype(np.int32), np.ceil(z).astype(np.int32)
        
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)
        z0 = np.clip(z0, 0, h - 1)
        z1 = np.clip(z1, 0, h - 1)

        h0 = self.data[z0, x0]
        h1 = self.data[z0, x1]
        h2 = self.data[z1, x0]
        h3 = self.data[z1, x1]
        H  = (h0 * (1 - a0) + h1 * a0) * (1 - a1) + (h2 * (1 - a0) + h3 * a0) * a1
        return self.v_scale * (H - self.offset)