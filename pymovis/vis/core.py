from __future__ import annotations

import numpy as np
from OpenGL.GL import *
import glm

class VAO:
    def __init__(self, id=None, vbos=None, ebo=None, indices=None) -> None:
        self.id      = id
        self.vbos    = vbos
        self.ebo     = ebo
        self.indices = indices

    @classmethod
    def from_vertex_array(cls, vertex_array: list[VertexGL], indices) -> VAO:
        """
        Constructor from a list of vertices and indices
        """
        id      = glGenVertexArrays(1)
        vbos    = glGenBuffers(8)
        ebo     = glGenBuffers(1)
        indices = indices

        glBindVertexArray(id)
        # glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # glBufferData(GL_ARRAY_BUFFER, VertexGL.sizeof(), vertex_array, GL_STATIC_DRAW)
        # glBufferData(GL_ARRAY_BUFFER, VertexGL.sizeof() * len(vertex_array), vertex_array, GL_STATIC_DRAW)

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

        # skinning indices
        data = np.array([v.skinning_indices1 for v in vertex_array], dtype=np.int32).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, vbos[4])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(4)
        glVertexAttribIPointer(4, 4, GL_INT, 0, ctypes.c_void_p(0))

        # skinning weights
        data = np.array([v.skinning_weights1 for v in vertex_array]).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, vbos[5])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(5)
        glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # skinning indices
        data = np.array([v.skinning_indices2 for v in vertex_array], dtype=np.int32).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, vbos[6])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(6)
        glVertexAttribIPointer(6, 4, GL_INT, 0, ctypes.c_void_p(0))

        # skinning weights
        data = np.array([v.skinning_weights2 for v in vertex_array]).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, vbos[7])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(7)
        glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

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

class VertexGL:
    def __init__(self, position=glm.vec3(0), normal=glm.vec3(0), uv=glm.vec2(0), material_id=0, skinning_indices1=glm.vec4(0), skinning_weights1=glm.vec4(0), skinning_indices2=glm.vec4(0), skinning_weights2=glm.vec4(0)):
        self.position         = position
        self.normal           = normal
        self.uv               = uv
        self.material_id      = material_id
        self.skinning_indices1 = skinning_indices1
        self.skinning_weights1 = skinning_weights1
        self.skinning_indices2 = skinning_indices2
        self.skinning_weights2 = skinning_weights2

    @staticmethod
    def make_vertex_array(positions, normals, tex_coords, lbs_indices1=None, lbs_weights1=None, lbs_indices2=None, lbs_weights2=None):
        vertex_array = []
        for i in range(len(positions)):
            if lbs_indices2 is not None and lbs_weights2 is not None:
                v = VertexGL(positions[i], normals[i], tex_coords[i], 0, lbs_indices1[i], lbs_weights1[i], lbs_indices2[i], lbs_weights2[i])
            elif lbs_indices1 is not None and lbs_weights1 is not None:
                v = VertexGL(positions[i], normals[i], tex_coords[i], 0, lbs_indices1[i], lbs_weights1[i])
            else:
                v = VertexGL(positions[i], normals[i], tex_coords[i], 0)
            vertex_array.append(v)
        return vertex_array

class MeshGL:
    def __init__(self, vao=None, vertices=None, indices=None):
        self.vao                = vao
        self.vertices           = vertices
        self.indices            = indices

        self.is_skinned         = False
        self.joint_order        = []
        self.name_to_idx        = {}
        self.joint_bind_trf_inv = []
    
    def generate_vertices(self):
        pass