from __future__ import annotations

import numpy as np
from OpenGL.GL import *
import glm

class VAO:
    def __init__(self, vertex_array: list[Vertex], indices) -> None:
        self.__id          = glGenVertexArrays(1)
        self.__vbos        = glGenBuffers(3)
        self.__ebo         = glGenBuffers(1)
        self.indices_count = len(indices)

        glBindVertexArray(self.__id)
        # glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # glBufferData(GL_ARRAY_BUFFER, Vertex.sizeof(), vertex_array, GL_STATIC_DRAW)
        # glBufferData(GL_ARRAY_BUFFER, Vertex.sizeof() * len(vertex_array), vertex_array, GL_STATIC_DRAW)

        # position
        data = np.array([v.position for v in vertex_array]).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, self.__vbos[0])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # normal
        data = np.array([v.normal for v in vertex_array]).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, self.__vbos[1])
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
        glBindBuffer(GL_ARRAY_BUFFER, self.__vbos[2])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # indices
        indices = np.array(indices, dtype=np.uint32)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.__ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    @property
    def id(self):
        return self.__id

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

    def __get_vertices(self):
        pass