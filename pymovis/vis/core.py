from __future__ import annotations

import numpy as np
from OpenGL.GL import *
import glm

def compute_tangent_space(vertices: list[VertexGL], indices: list[int]) -> list[VertexGL]:
    # https://learnopengl.com/Advanced-Lighting/Normal-Mapping

    if len(vertices) == 0 or len(indices) == 0:
        raise Exception("Empty vertex array or index array")
    
    if len(indices) % 3 != 0:
        raise Exception("Index array length must be a multiple of 3")

    tangents, bitangents = [], []
    for i in range(0, len(indices), 3):
        delta_pos1 = glm.vec3(vertices[indices[i+1]].position - vertices[indices[i]].position)
        delta_pos2 = glm.vec3(vertices[indices[i+2]].position - vertices[indices[i]].position)

        delta_uv1 = glm.vec2(vertices[indices[i+1]].uv - vertices[indices[i]].uv)
        delta_uv2 = glm.vec2(vertices[indices[i+2]].uv - vertices[indices[i]].uv)

        det = 1.0 / ((delta_uv1.x * delta_uv2.y - delta_uv2.x * delta_uv1.y) + 1e-8)

        tangent = glm.normalize(det * (delta_uv2.y * delta_pos1 - delta_uv1.y * delta_pos2))
        bitangent = glm.normalize(det * (-delta_uv2.x * delta_pos1 + delta_uv1.x * delta_pos2))

        tangents.append(tangent)
        bitangents.append(bitangent)
    
    for i in range(0, len(indices), 3):
        idx = i//3
        
        vertices[indices[i]].set_tangent(tangents[idx])
        vertices[indices[i]].set_bitangent(bitangents[idx])

        vertices[indices[i+1]].set_tangent(tangents[idx])
        vertices[indices[i+1]].set_bitangent(bitangents[idx])

        vertices[indices[i+2]].set_tangent(tangents[idx])
        vertices[indices[i+2]].set_bitangent(bitangents[idx])

    return vertices

class VAO:
    def __init__(self, id=None, vbos=None, ebo=None, indices=None) -> None:
        self.id      = id
        self.vbos    = vbos
        self.ebo     = ebo
        self.indices = indices

    @classmethod
    def from_vertex_array(cls, vertex_array: list[VertexGL], indices, compute_tangent=True) -> VAO:
        # compute tangent and bitangent
        if compute_tangent:
            vertex_array = compute_tangent_space(vertex_array, indices)

        id   = glGenVertexArrays(1)
        vbos = glGenBuffers(10)
        ebo  = glGenBuffers(1)

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

        # tex_coord
        data = np.array([v.uv for v in vertex_array]).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, vbos[2])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # tangent
        data = np.array([v.tangent for v in vertex_array]).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, vbos[3])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # bitangent
        data = np.array([v.bitangent for v in vertex_array]).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, vbos[4])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # material id
        data = np.array([v.material_id for v in vertex_array], dtype=np.int32).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, vbos[5])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(5)
        glVertexAttribIPointer(5, 1, GL_INT, 0, ctypes.c_void_p(0))

        # skinning indices
        data = np.array([v.skinning_indices1 for v in vertex_array], dtype=np.int32).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, vbos[6])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(6)
        glVertexAttribIPointer(6, 4, GL_INT, 0, ctypes.c_void_p(0))

        # skinning weights
        data = np.array([v.skinning_weights1 for v in vertex_array]).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, vbos[7])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(7)
        glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # skinning indices
        data = np.array([v.skinning_indices2 for v in vertex_array], dtype=np.int32).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, vbos[8])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(8)
        glVertexAttribIPointer(8, 4, GL_INT, 0, ctypes.c_void_p(0))

        # skinning weights
        data = np.array([v.skinning_weights2 for v in vertex_array]).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, vbos[9])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(9)
        glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # indices
        data = np.array(indices, dtype=np.uint32).flatten()
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        # unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        return cls(id, vbos, ebo, indices)
    
    @classmethod
    def from_positions(cls, positions) -> VAO:
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
    def __init__(
        self,
        position         : glm.vec3 = glm.vec3(0),
        normal           : glm.vec3 = glm.vec3(0),
        uv               : glm.vec2 = glm.vec2(0),
        tangent          : glm.vec3 = glm.vec3(0),
        bitangent        : glm.vec3 = glm.vec3(0),
        material_id      : int      = 0,
        skinning_indices1: glm.vec4 = glm.vec4(0),
        skinning_weights1: glm.vec4 = glm.vec4(0),
        skinning_indices2: glm.vec4 = glm.vec4(0),
        skinning_weights2: glm.vec4 = glm.vec4(0)
    ):
        self.position          = position
        self.normal            = normal
        self.uv                = uv
        self.tangent           = tangent
        self.bitangent         = bitangent
        self.material_id       = material_id
        self.skinning_indices1 = skinning_indices1
        self.skinning_weights1 = skinning_weights1
        self.skinning_indices2 = skinning_indices2
        self.skinning_weights2 = skinning_weights2

    @staticmethod
    def make_vertex_array(positions, normals, tex_coords, tangents=None, bitangents=None, lbs_indices1=None, lbs_weights1=None, lbs_indices2=None, lbs_weights2=None) -> list[VertexGL]:
        vertex_array = []
        for i in range(len(positions)):
            tangent = glm.vec3(0) if tangents is None else tangents[i]
            bitangent = glm.vec3(0) if bitangents is None else bitangents[i]

            if lbs_indices2 is not None and lbs_weights2 is not None:
                v = VertexGL(positions[i], normals[i], tex_coords[i], tangent, bitangent, 0, lbs_indices1[i], lbs_weights1[i], lbs_indices2[i], lbs_weights2[i])
            elif lbs_indices1 is not None and lbs_weights1 is not None:
                v = VertexGL(positions[i], normals[i], tex_coords[i], tangent, bitangent, 0, lbs_indices1[i], lbs_weights1[i])
            else:
                v = VertexGL(positions[i], normals[i], tex_coords[i], tangent, bitangent, 0)
            vertex_array.append(v)
        
        return vertex_array
    
    def set_tangent(self, tangent):
        self.tangent = tangent

    def set_bitangent(self, bitangent):
        self.bitangent = bitangent

class MeshGL:
    def __init__(self, vao=None, vertices=None, indices=None):
        self.vao                = vao
        self.vertices           = vertices
        self.indices            = indices

        self.is_skinned         = False
        self.joint_order        = []
        self.name_to_idx        = {}
        self.joint_bind_xform_inv = []
    
    def generate_vertices(self):
        pass