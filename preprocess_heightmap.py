from OpenGL.GL import *
import glm
import math
import numpy as np

from pymovis.vis.core import Vertex, VAO, Mesh
from pymovis.vis.app import App
from pymovis.vis.appmanager import AppManager
from pymovis.vis.render import Render

class Heightmap:
    def __init__(self, filename, h_scale=0.1, v_scale=0.1, offset=None):
        self.filename = filename
        self.h_scale = h_scale
        self.v_scale = v_scale
        self.offset = offset
    
    def load(self):
        """ Load the heightmap data and create the vertices """
        self.data = np.loadtxt(self.filename, dtype=np.float32)
        w = len(self.data)
        h = len(self.data[0])

        vertices = [Vertex() for _ in range(w * h)]

        """ Calculate the offset """
        self.offset = np.sum(self.data) / (w * h) if self.offset is None else 0
        print(f"Loaded Heightmap {self.filename} with {w}x{h} points")

        """ Calculate and set the positions of the vertices """
        cw = self.h_scale * w
        ch = self.h_scale * h
        cx = self.h_scale * np.arange(w)
        cy = self.h_scale * np.arange(h)
        cx, cy = np.meshgrid(cx, cy)

        x_pos = cx - cw / 2
        z_pos = cy - ch / 2
        y_pos = self.sample(x_pos, z_pos)
        positions = np.stack([x_pos, y_pos, z_pos], axis=-1)

        for i, pos in enumerate(positions.reshape(-1, 3)):
            vertices[i].set_position(pos)
        
        """ Calculate and set the normals of the vertices """
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
        
        """ Set vertex indices """
        indices = np.empty((h - 1, w - 1, 6), dtype=np.int32)
        indices[..., 0] = np.arange(h * w).reshape(h, w)[:-1, :-1]
        indices[..., 1] = indices[..., 0] + w
        indices[..., 2] = indices[..., 0] + 1
        indices[..., 3] = indices[..., 0] + 1
        indices[..., 4] = indices[..., 0] + w
        indices[..., 5] = indices[..., 0] + w + 1
        indices = indices.flatten()

        """ Generate VAO and Mesh """
        vao = VAO.from_vertex_array(vertices, indices)
        self.mesh = Mesh(vao, vertices, indices)

    def sample(self, x, z):
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

""" Global variables """
HEIGHTMAP_PATH = './data/heightmaps/hmap_011_smooth.txt'

class HeightmapApp(App):
    def __init__(self, heightmap: Heightmap):
        super().__init__()
        import time
        start = time.perf_counter()
        heightmap.load()
        print(f"Loaded heightmap in {time.perf_counter() - start} seconds")
        self.heightmap = Render.mesh(heightmap.mesh).set_material(albedo=glm.vec3(0.5))
        self.axis = Render.axis()
        # self.grid = Render.plane().set_scale(10).set_texture("grid.png")
    
    def render(self):
        self.heightmap.draw()
        # self.axis.draw()

if __name__ == "__main__":
    heightmap = Heightmap(HEIGHTMAP_PATH, 0.01, 0.01)

    app_manager = AppManager()
    app = HeightmapApp(heightmap)
    app_manager.run(app)