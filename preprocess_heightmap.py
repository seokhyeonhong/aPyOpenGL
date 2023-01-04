from OpenGL.GL import *
import glm
import math
import numpy as np

from pymovis.vis.core import Vertex, VAO, Mesh
from pymovis.vis.app import App
from pymovis.vis.appmanager import AppManager
from pymovis.vis.render import Render

class Heightmap:
    def __init__(self, filename, h_scale=0.1, v_scale=0.1):
        self.filename = filename
        self.h_scale = h_scale
        self.v_scale = v_scale
    
    def load(self):
        self.data = np.loadtxt(self.filename, dtype=np.float32)
        w = len(self.data)
        h = len(self.data[0])
        self.offset = np.sum(self.data) / (w * h)
        print(f"Loaded Heightmap {self.filename} with {w}x{h} points")

        vertices = [Vertex() for _ in range(w * h)]

        for x in range(w):
            for y in range(h):
                cx = self.h_scale * x
                cy = self.h_scale * y
                cw = self.h_scale * w
                ch = self.h_scale * h
                vertices[y * w + x].set_position(glm.vec3(cx - cw / 2, self.sample(glm.vec2(cx - cw / 2, cy - ch / 2)), cy - ch / 2))
        
        for x in range(w):
            for y in range(h):
                if x > 0 and x < w - 1 and y > 0 and y < h - 1:
                    vertices[y * w + x].set_normal(glm.normalize(glm.mix(
                        glm.cross(
                            vertices[(x+0) + (y+1) * w].position - vertices[x+y*w].position,
                            vertices[(x+1) + (y+0) * w].position - vertices[x+y*w].position),
                        glm.cross(
                            vertices[(x+0) + (y-1) * w].position - vertices[x+y*w].position,
                            vertices[(x-1) + (y+0) * w].position - vertices[x+y*w].position), 0.5)))
                else:
                    vertices[y * w + x].set_normal(glm.vec3(0, 1 , 0))
        
        indices = [None] * (w - 1) * (h - 1) * 6
        for x in range(w - 1):
            for y in range(h - 1):
                indices[(x + y * (w - 1)) * 6 + 0] = x + y * w
                indices[(x + y * (w - 1)) * 6 + 1] = x + (y + 1) * w
                indices[(x + y * (w - 1)) * 6 + 2] = x + 1 + y * w
                indices[(x + y * (w - 1)) * 6 + 3] = x + 1 + y * w
                indices[(x + y * (w - 1)) * 6 + 4] = x + (y + 1) * w
                indices[(x + y * (w - 1)) * 6 + 5] = x + 1 + (y + 1) * w

        vao = VAO.from_vertex_array(vertices, indices)
        self.mesh = Mesh(vao, vertices, indices)

    def sample(self, pos: glm.vec2):
        w = len(self.data)
        h = len(self.data[0])

        pos.x = (pos.x / self.h_scale) + (w / 2)
        pos.y = (pos.y / self.h_scale) + (h / 2)

        a0 = math.fmod(pos.x, 1)
        a1 = math.fmod(pos.y, 1)

        x0, x1 = math.floor(pos.x), math.ceil(pos.x)
        y0, y1 = math.floor(pos.y), math.ceil(pos.y)

        x0 = min(w - 1, max(0, x0))
        x1 = min(w - 1, max(0, x1))
        y0 = min(h - 1, max(0, y0))
        y1 = min(h - 1, max(0, y1))

        s0 = self.v_scale * (self.data[x0][y0] - self.offset)
        s1 = self.v_scale * (self.data[x1][y0] - self.offset)
        s2 = self.v_scale * (self.data[x0][y1] - self.offset)
        s3 = self.v_scale * (self.data[x1][y1] - self.offset)

        return (s0 * (1 - a0) + s1 * a0) * (1 - a1) + (s2 * (1 - a0) + s3 * a0) * a1

""" Global variables """
HEIGHTMAP_PATH = './data/heightmaps/hmap_012_smooth.txt'

class HeightmapApp(App):
    def __init__(self, heightmap: Heightmap):
        super().__init__()
        self.heightmap = heightmap
        self.heightmap.load()
    
    def render(self):
        Render.mesh(self.heightmap.mesh).set_material(albedo=glm.vec3(0.5)).draw()

if __name__ == "__main__":
    heightmap = Heightmap(HEIGHTMAP_PATH)

    app_manager = AppManager()
    app = HeightmapApp(heightmap)
    app_manager.run(app)