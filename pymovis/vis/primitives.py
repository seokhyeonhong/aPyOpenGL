import glm
import numpy as np

from pymovis.vis.core import VAO, Vertex, Mesh

def get_color_by_position(position):
    colors = []
    for p in position:
        normalized = glm.normalize(p)
        colors.append(normalized)
    return colors
"""
   v2----- v1
  /        / 
 v3------v0
"""
class Plane(Mesh):
    def __init__(self):
        positions, normals, tex_coords, indices = self.get_vertices()
        vertices = Vertex.make_vertex_array(positions, normals, tex_coords)
        vao = VAO.from_vertex_array(vertices, indices)
        super().__init__(vao, vertices, indices)

    def get_vertices(self):
        positions = [
            glm.vec3(0.5, 0.0, 0.5),   # v0
            glm.vec3(0.5, 0.0, -0.5),  # v1
            glm.vec3(-0.5, 0.0, -0.5), # v2
            glm.vec3(-0.5, 0.0, 0.5),  # v3
        ]
        normals = [glm.vec3(0.0, 1.0, 0.0)] * 4
        tex_coords = [
            glm.vec2(1.0, 0.0),
            glm.vec2(1.0, 1.0),
            glm.vec2(0.0, 1.0),
            glm.vec2(0.0, 0.0),
        ]
        indices = [0, 1, 2, 2, 3, 0]
        return positions, normals, tex_coords, indices
"""
   v6----- v5
  /|      /| 
 v1------v0| 
 | |     | | 
 | v7----|-v4
 |/      |/  
 v2------v3  
"""
class Cube(Mesh):
    def __init__(self):
        positions, normals, tex_coords, indices = self.get_vertices()
        vertices = Vertex.make_vertex_array(positions, normals, tex_coords)
        vao = VAO.from_vertex_array(vertices, indices)
        super().__init__(vao, vertices, indices)

    def get_vertices(self):
        v = [
            glm.vec3(0.5, 0.5, 0.5), glm.vec3(-0.5, 0.5, 0.5), glm.vec3(-0.5, -0.5, 0.5), glm.vec3(0.5, -0.5, 0.5),
            glm.vec3(0.5, -0.5, -0.5), glm.vec3(0.5, 0.5, -0.5), glm.vec3(-0.5, 0.5, -0.5), glm.vec3(-0.5, -0.5, -0.5)
        ]
        n = [
            glm.vec3(0.0, 0.0, 1.0), glm.vec3(1.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0),
            glm.vec3(-1.0, 0.0, 0.0), glm.vec3(0.0, -1.0, 0.0), glm.vec3(0.0, 0.0, -1.0)
        ]
        t = [
            glm.vec2(1.0, 1.0), glm.vec2(0.0, 1.0), glm.vec2(0.0, 0.0), glm.vec2(1.0, 0.0)
        ]

        positions = [
            v[0], v[1], v[2], v[3], # front
            v[0], v[3], v[4], v[5], # right
            v[0], v[5], v[6], v[1], # top
            v[1], v[6], v[7], v[2], # left
            v[7], v[4], v[3], v[2], # bottom
            v[4], v[7], v[6], v[5]  # back
        ]

        normals = [
            n[0], n[0], n[0], n[0], # front
            n[1], n[1], n[1], n[1], # right
            n[2], n[2], n[2], n[2], # top
            n[3], n[3], n[3], n[3], # left
            n[4], n[4], n[4], n[4], # bottom
            n[5], n[5], n[5], n[5]  # back
        ]

        tex_coords = [
            t[0], t[1], t[2], t[3], # front
            t[1], t[2], t[3], t[0], # right
            t[3], t[0], t[1], t[2], # top
            t[0], t[1], t[2], t[3], # left
            t[2], t[3], t[0], t[1], # bottom
            t[2], t[3], t[0], t[1]  # back
        ]

        indices = [
            0, 1, 2, 2, 3, 0,       # front
            4, 5, 6, 6, 7, 4,       # right
            8, 9, 10, 10, 11, 8,    # top
            12, 13, 14, 14, 15, 12, # left
            16, 17, 18, 18, 19, 16, # bottom
            20, 21, 22, 22, 23, 20  # back
        ]

        # self.line_indices = [
        #     0, 1, 1, 2, 2, 3, 3, 0, # front
        #     4, 5, 5, 6, 6, 7, 7, 4, # right
        #     8, 9, 9, 10, 10, 11, 11, 8, # top
        #     12, 13, 13, 14, 14, 15, 15, 12, # left
        #     16, 17, 17, 18, 18, 19, 19, 16, # bottom
        #     20, 21, 21, 22, 22, 23, 23, 20  # back
        # ]
        return positions, normals, tex_coords, indices

class Sphere(Mesh):
    def __init__(
        self,
        radius,
        stacks=16,
        sectors=16
    ):
        positions, normals, tex_coords, indices = self.get_vertices(radius, stacks, sectors)
        vertices = Vertex.make_vertex_array(positions, normals, tex_coords)
        vao = VAO.from_vertex_array(vertices, indices)
        super().__init__(vao, vertices, indices)

    def get_vertices(self, radius, stacks, sectors):
        # TODO: Optimization using numpy (e.g. without for loops)
        """
        theta: angle between up-axis(=y) and the point on the sphere
        phi: angle between the projection of the point on horizontal plane(=xz) and forward axis(=z)
        """
        positions, normals, tex_coords = [], [], []
        for i in range(stacks+1):
            theta = i * np.pi / stacks
            y = radius * np.cos(theta)
            r_sin_theta = radius * np.sin(theta)
            for j in range(sectors+1):
                phi = j * 2 * np.pi / sectors
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)

                x = r_sin_theta * sin_phi
                z = r_sin_theta * cos_phi

                positions.append(glm.vec3(x, y, z))
                normals.append(glm.normalize(glm.vec3(x, y, z)))
                tex_coords.append(glm.vec2(j / sectors, i / stacks))

        indices = []
        for i in range(stacks):
            for j in range(sectors):
                p1 = i * (sectors + 1) + j
                p2 = p1 + sectors + 1

                if i != 0:
                    indices.append(p1)
                    indices.append(p2)
                    indices.append(p1+1)

                if i != (stacks-1):
                    indices.append(p1 + 1)
                    indices.append(p2)
                    indices.append(p2 + 1)

        return positions, normals, tex_coords, indices

class Cone(Mesh):
    def __init__(
        self,
        radius,
        height,
        sectors=16
    ):
        positions, normals, tex_coords, indices = self.get_vertices(radius, height, sectors)
        vertices = Vertex.make_vertex_array(positions, normals, tex_coords)
        vao = VAO.from_vertex_array(vertices, indices)
        super().__init__(vao, vertices, indices)
    
    def get_vertices(self, radius, height, sectors):
        positions, normals, tex_coords = [], [], []
        indices = []

        half_height = height * 0.5

        # side 
        positions.append(glm.vec3(0, half_height, 0))
        normals.append(glm.vec3(0, 1, 0))
        tex_coords.append(glm.vec2(0.5, 0.5))

        theta = np.linspace(0, 2 * np.pi, sectors+1)
        x = radius * np.sin(theta)
        z = radius * np.cos(theta)

        for i in range(sectors+1):
            positions.append(glm.vec3(x[i], -half_height, z[i]))
            normals.append(glm.vec3(x[i] / radius, 0, z[i] / radius))
            tex_coords.append(glm.vec2(z[i] / radius * 0.5 + 0.5, x[i] / radius * 0.5 + 0.5))

            if i < sectors:
                indices.append(0)
                indices.append(i+1)
                indices.append(i+2)

        # bottom
        positions.append(glm.vec3(0, -half_height, 0))
        normals.append(glm.vec3(0, -1, 0))
        tex_coords.append(glm.vec2(0.5, 0.5))

        for i in range(sectors+1):
            positions.append(glm.vec3(x[i], -half_height, z[i]))
            normals.append(glm.vec3(0, -1, 0))
            tex_coords.append(glm.vec2(z[i] / radius * 0.5 + 0.5, x[i] / radius * 0.5 + 0.5))

            if i < sectors:
                indices.append(sectors+2)
                indices.append(sectors+4+i)
                indices.append(sectors+3+i)

        return positions, normals, tex_coords, indices

class Cylinder(Mesh):
    def __init__(
        self,
        radius,
        height,
        sectors=16
    ):
        positions, normals, tex_coords, indices = self.get_vertices(radius, height, sectors)
        vertices = Vertex.make_vertex_array(positions, normals, tex_coords)
        vao = VAO.from_vertex_array(vertices, indices)
        super().__init__(vao, vertices, indices)
    
    def get_vertices(self, radius, height, sectors):
        positions, normals, tex_coords = [], [], []
        indices = []

        half_height = height * 0.5
        theta = np.linspace(0, 2 * np.pi, sectors+1)
        x = radius * np.sin(theta)
        z = radius * np.cos(theta)

        # top
        positions.append(glm.vec3(0, half_height, 0))
        normals.append(glm.vec3(0, 1, 0))
        tex_coords.append(glm.vec2(0.5, 0.5))
        
        index_offset = len(positions)
        for i in range(sectors+1):
            positions.append(glm.vec3(x[i], half_height, z[i]))
            normals.append(glm.vec3(0, 1, 0))
            tex_coords.append(glm.vec2(z[i] / radius * 0.5 + 0.5, x[i] / radius * 0.5 + 0.5))

            if i < sectors:
                indices.append(index_offset - 1)
                indices.append(index_offset + i)
                indices.append(index_offset + i + 1)
        
        # side
        index_offset = len(positions)
        for i in range(sectors+1):
            positions.append(glm.vec3(x[i], half_height, z[i]))
            normals.append(glm.vec3(x[i] / radius, 0, z[i] / radius))
            tex_coords.append(glm.vec2(i / sectors, 0))

            positions.append(glm.vec3(x[i], -half_height, z[i]))
            normals.append(glm.vec3(x[i] / radius, 0, z[i] / radius))
            tex_coords.append(glm.vec2(i / sectors, 1))

            if i < sectors:
                indices.append(index_offset + i * 2)
                indices.append(index_offset + i * 2 + 1)
                indices.append(index_offset + i * 2 + 2)

                indices.append(index_offset + i * 2 + 1)
                indices.append(index_offset + i * 2 + 3)
                indices.append(index_offset + i * 2 + 2)
        
        # bottom
        positions.append(glm.vec3(0, -half_height, 0))
        normals.append(glm.vec3(0, -1, 0))
        tex_coords.append(glm.vec2(0.5, 0.5))

        index_offset = len(positions)
        for i in range(sectors+1):
            positions.append(glm.vec3(x[i], -half_height, z[i]))
            normals.append(glm.vec3(0, -1, 0))
            tex_coords.append(glm.vec2(z[i] / radius * 0.5 + 0.5, x[i] / radius * 0.5 + 0.5))

            if i < sectors:
                indices.append(index_offset - 1)
                indices.append(index_offset + i + 1)
                indices.append(index_offset + i)
        
        return positions, normals, tex_coords, indices

class Cubemap(Mesh):
    """
    Cubemap is actually not a mesh itself, but this implementation is convenient for rendering.
    """
    def __init__(self, scale=100):
        positions = self.get_vertices()
        positions = np.array(positions, dtype=np.float32) * scale
        vao = VAO.from_positions(positions)
        super().__init__(vao, None, None)
    
    def get_vertices(self):
        positions = [
            glm.vec3(-1,  1, -1),
            glm.vec3(-1, -1, -1),
            glm.vec3( 1, -1, -1),
            glm.vec3( 1, -1, -1),
            glm.vec3( 1,  1, -1),
            glm.vec3(-1,  1, -1),

            glm.vec3(-1, -1,  1),
            glm.vec3(-1, -1, -1),
            glm.vec3(-1,  1, -1),
            glm.vec3(-1,  1, -1),
            glm.vec3(-1,  1,  1),
            glm.vec3(-1, -1,  1),

            glm.vec3( 1, -1, -1),
            glm.vec3( 1, -1,  1),
            glm.vec3( 1,  1,  1),
            glm.vec3( 1,  1,  1),
            glm.vec3( 1,  1, -1),
            glm.vec3( 1, -1, -1),

            glm.vec3(-1, -1,  1),
            glm.vec3(-1,  1,  1),
            glm.vec3( 1,  1,  1),
            glm.vec3( 1,  1,  1),
            glm.vec3( 1, -1,  1),
            glm.vec3(-1, -1,  1),

            glm.vec3(-1,  1, -1),
            glm.vec3( 1,  1, -1),
            glm.vec3( 1,  1,  1),
            glm.vec3( 1,  1,  1),
            glm.vec3(-1,  1,  1),
            glm.vec3(-1,  1, -1),

            glm.vec3(-1, -1, -1),
            glm.vec3(-1, -1,  1),
            glm.vec3( 1, -1, -1),
            glm.vec3( 1, -1, -1),
            glm.vec3(-1, -1,  1),
            glm.vec3( 1, -1,  1)
        ]
        return positions