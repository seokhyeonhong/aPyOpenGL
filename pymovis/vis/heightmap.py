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
        """ Create vertices from the heightmap data """
        w = len(self.data)
        h = len(self.data[0])

        self.offset = np.sum(self.data) / (w * h) if self.offset is None else 0
        print(f"Loaded Heightmap: {w}x{h} points ({self.h_scale * w:.4f}m x {self.h_scale * h:.4f}m)")

        vertices = [Vertex() for _ in range(w * h)]

        """ Vertex positions """
        cw = self.h_scale * w
        ch = self.h_scale * h
        cx = self.h_scale * np.arange(w)
        cy = self.h_scale * np.arange(h)
        cx, cy = np.meshgrid(cx, cy)

        x_pos = cx - cw / 2
        z_pos = cy - ch / 2
        y_pos = self.sample_height(x_pos, z_pos)
        positions = np.stack([x_pos, y_pos, z_pos], axis=-1)

        for i, pos in enumerate(positions.reshape(-1, 3)):
            vertices[i].set_position(pos)
        
        """ Vertex normals """
        normals = np.empty((h, w, 3), dtype=np.float32)

        cross1 = np.cross(positions[2:, 1:-1] - positions[1:-1, 1:-1], positions[1:-1, 2:] - positions[1:-1, 1:-1])
        cross2 = np.cross(positions[:-2, 1:-1] - positions[1:-1, 1:-1], positions[1:-1, :-2] - positions[1:-1, 1:-1])
        cross = (cross1 + cross2) * 0.5
        cross = cross / (np.linalg.norm(cross, axis=-1, keepdims=True) + 1e-8)

        normals[1:-1, 1:-1] = cross
        normals[0, :] = normals[-1, :] = np.array([0, 1, 0])
        normals[:, 0] = normals[:, -1] = np.array([0, 1, 0])

        for i, normal in enumerate(normals.reshape(-1, 3)):
            vertices[i].set_normal(normal)
        
        """ Vertex UV coordinates """
        uvs = np.empty((h, w, 2), dtype=np.float32)

        uvs[:, :, 0] = cy
        uvs[:, :, 1] = cx

        for i, uv in enumerate(uvs.reshape(-1, 2)):
            vertices[i].set_uv(uv)

        """ Vertex indices """
        indices = np.empty((h - 1, w - 1, 6), dtype=np.int32)
        indices[..., 0] = np.arange(h * w).reshape(h, w)[:-1, :-1]
        indices[..., 1] = indices[..., 4] = indices[..., 0] + w
        indices[..., 2] = indices[..., 3] = indices[..., 0] + 1
        indices[..., 5] = indices[..., 0] + w + 1
        indices = indices.flatten()

        """ VAO and Mesh """
        vao = VAO.from_vertex_array(vertices, indices)
        self.mesh = Mesh(vao, vertices, indices)

    def sample_height(self, x, z):
        w = len(self.data)
        h = len(self.data[0])

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

        s0 = self.v_scale * (self.data[x0, z0] - self.offset)
        s1 = self.v_scale * (self.data[x1, z0] - self.offset)
        s2 = self.v_scale * (self.data[x0, z1] - self.offset)
        s3 = self.v_scale * (self.data[x1, z1] - self.offset)

        return (s0 * (1 - a0) + s1 * a0) * (1 - a1) + (s2 * (1 - a0) + s3 * a0) * a1