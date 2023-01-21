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
        id      = glGenVertexArrays(1)
        vbos    = glGenBuffers(4)
        ebo     = glGenBuffers(1)
        indices = indices

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

        # material id
        data = np.array([v.material_id for v in vertex_array], dtype=np.int32).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, vbos[3])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(3)
        glVertexAttribIPointer(3, 1, GL_INT, 0, ctypes.c_void_p(0))

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
        id      = glGenVertexArrays(1)
        vbos    = glGenBuffers(1)
        ebo     = None
        indices = None

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
    def id(self):
        return self.__id

    @property
    def len_indices(self):
        return len(self.__indices)

class Vertex:
    def __init__(self):
        self.position    = glm.vec3(0)
        self.normal      = glm.vec3(0)
        self.uv          = glm.vec2(0)
        self.material_id = 0

    @staticmethod
    def make_vertex_array(positions, normals, tex_coords):
        res = []
        for i in range(len(positions)):
            v = Vertex()
            v.position    = positions[i]
            v.normal      = normals[i]
            v.uv          = tex_coords[i]
            v.material_id = 0
            res.append(v)
        return res

class Mesh:
    def __init__(self, vao=None, vertices=None, indices=None):
        self.vao      = vao
        self.vertices: list[Vertex] = vertices
        self.indices  = indices

        self.is_skinned = False
        self.joint_order = []
        self.name_to_idx = {}
        self.joint_bind_trf_inv = []
    
    def generate_vertices(self):
        pass