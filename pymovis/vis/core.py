from __future__ import annotations

import numpy as np
from OpenGL.GL import *
import glm

class VAO:
    def __init__(self, id=None, vbos=None, ebo=None, indices=None) -> None:
        self.__id      = id
        self.__vbos    = vbos
        self.__ebo     = ebo
        self.__indices = indices

    @classmethod
    def from_vertex_array(cls, vertex_array: list[Vertex], indices) -> VAO:
        """
        Constructor from a list of vertices and indices
        """
        id          = glGenVertexArrays(1)
        vbos        = glGenBuffers(3)
        ebo         = glGenBuffers(1)
        indices     = indices

        glBindVertexArray(id)
        # glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # glBufferData(GL_ARRAY_BUFFER, Vertex.sizeof(), vertex_array, GL_STATIC_DRAW)
        # glBufferData(GL_ARRAY_BUFFER, Vertex.sizeof() * len(vertex_array), vertex_array, GL_STATIC_DRAW)

        # position
        data = np.array([v.position for v in vertex_array]).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, vbos[0])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # normal
        data = np.array([v.normal for v in vertex_array]).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, vbos[1])
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
        glBindBuffer(GL_ARRAY_BUFFER, vbos[2])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # indices
        indices = np.array(indices, dtype=np.uint32)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        return cls(id, vbos, ebo, indices)
    
    @classmethod
    def from_positions(cls, positions) -> VAO:
        """
        Constructor from a list of positions
        """
        id          = glGenVertexArrays(1)
        vbos        = glGenBuffers(1)
        ebo         = None
        indices     = None

        glBindVertexArray(id)
        data = np.array(positions).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, vbos)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        return cls(id, vbos, ebo, indices)

    @property
    def id(self) -> int:
        return self.__id

    @property
    def indices_count(self):
        return len(self.__indices)

class Vertex:
    def __init__(
        self,
        position:glm.vec3=glm.vec3(0),
        normal  :glm.vec3=glm.vec3(0),
        uv      :glm.vec2=glm.vec2(0)
    ):
        self.__position = position
        self.__normal   = normal
        self.__uv       = uv
    
    @property
    def position(self):
        return self.__position
    
    @property
    def normal(self):
        return self.__normal

    @property
    def uv(self):
        return self.__uv

    @staticmethod
    def make_vertex_array(positions, normals, tex_coords):
        res = []
        for i in range(len(positions)):
            res.append(Vertex(positions[i], normals[i], tex_coords[i]))
        return res


class Mesh:
    def __init__(
        self,
        vao     : VAO,
        vertices: list[Vertex],
        indices : list[int]
    ):
        self.__vao      = vao
        self.__vertices = vertices
        self.__indices  = indices
    
    @property
    def vao(self):
        return self.__vao
    
    def get_vertices(self):
        pass