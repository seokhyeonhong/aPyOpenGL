import numpy as np
from OpenGL.GL import *
import glm

def to_vertex_array(positions, normals, tex_coords):
    res = []
    for i in range(len(positions)):
        res.append(Vertex(positions[i], normals[i], tex_coords[i]))
    return res

class VAO:
    def __init__(self, vertex_array, indices):
        self.vao_id        = glGenVertexArrays(1)
        self.vbos          = glGenBuffers(3)
        self.ebo           = glGenBuffers(1)
        self.indices_count = len(indices)

        glBindVertexArray(self.vao_id)
        # glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # glBufferData(GL_ARRAY_BUFFER, Vertex.sizeof(), vertex_array, GL_STATIC_DRAW)
        # glBufferData(GL_ARRAY_BUFFER, Vertex.sizeof() * len(vertex_array), vertex_array, GL_STATIC_DRAW)

        # position
        data = np.array([v.position for v in vertex_array]).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, self.vbos[0])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # normal
        data = np.array([v.normal for v in vertex_array]).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, self.vbos[1])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # # color - deprecated
        # data = np.array([v.color for v in vertex_array]).flatten()
        # glBindBuffer(GL_ARRAY_BUFFER, self.vbos[2])
        # glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        # glEnableVertexAttribArray(2)
        # glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # tex_coord
        data = np.array([v.uv for v in vertex_array]).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, self.vbos[2])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # indices
        indices = np.array(indices, dtype=np.uint32)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

class Vertex:
    def __init__(
        self,
        position:glm.vec3=glm.vec3(0),
        normal  :glm.vec3=glm.vec3(0),
        uv      :glm.vec2=glm.vec2(0)
    ):
        self.position = position
        self.normal   = normal
        self.uv       = uv
    
    # @staticmethod
    # def sizeof():
    #     return glm.sizeof(glm.vec3) + glm.sizeof(glm.vec3)
    
    # @staticmethod
    # def offsetof(field):
    #     offset_dict = {
    #         "position": 0,
    #         "normal": glm.sizeof(glm.vec3),
    #     }
    #     return ctypes.c_void_p(offset_dict[field])

class Mesh:
    def __init__(
        self,
        vao     : VAO,
        vertices: list, # Vertex
        indices : list  # int
    ):
        self.vao      = vao
        self.vertices = vertices
        self.indices  = indices