import glm
import numpy as np

from pymovis.vis import core

def get_color_by_position(position):
    colors = []
    for p in position:
        normalized = glm.normalize(p)
        colors.append(normalized)
    return colors

# plane
#    v2----- v1
#   /       /   
#  v3------v0
class Plane(core.Mesh):
    def __init__(self):
        self.positions = [
            glm.vec3(0.5, 0.0, 0.5),   # v0
            glm.vec3(0.5, 0.0, -0.5),  # v1
            glm.vec3(-0.5, 0.0, -0.5), # v2
            glm.vec3(-0.5, 0.0, 0.5),  # v3
        ]
        self.normals = [glm.vec3(0.0, 1.0, 0.0)] * 4
        self.tex_coords = [
            glm.vec2(1.0, 0.0),
            glm.vec2(1.0, 1.0),
            glm.vec2(0.0, 1.0),
            glm.vec2(0.0, 0.0),
        ]
        self.indices = [0, 1, 2, 2, 3, 0]
        # self.line_indices = [0, 1, 1, 2, 2, 3, 3, 0]
        vertices = core.to_vertex_array(self.positions, self.normals, self.tex_coords)
        vao = core.VAO(vertices, self.indices)
        super().__init__(vao, vertices, self.indices)

class Cube(core.Mesh):
    #    v6----- v5  
    #   /|      /|   
    #  v1------v0|   
    #  | |     | |   
    #  | v7----|-v4  
    #  |/      |/    
    #  v2------v3    
    def __init__(self):
        v = [
            glm.vec3(0.5, 0.5, 0.5),
            glm.vec3(-0.5, 0.5, 0.5),
            glm.vec3(-0.5, -0.5, 0.5),
            glm.vec3(0.5, -0.5, 0.5),
            glm.vec3(0.5, -0.5, -0.5),
            glm.vec3(0.5, 0.5, -0.5),
            glm.vec3(-0.5, 0.5, -0.5),
            glm.vec3(-0.5, -0.5, -0.5)
        ]
        n = [
            glm.vec3(0.0, 0.0, 1.0),
            glm.vec3(1.0, 0.0, 0.0),
            glm.vec3(0.0, 1.0, 0.0),
            glm.vec3(-1.0, 0.0, 0.0),
            glm.vec3(0.0, -1.0, 0.0),
            glm.vec3(0.0, 0.0, -1.0)
        ]
        t = [
            glm.vec2(1.0, 1.0),
            glm.vec2(0.0, 1.0),
            glm.vec2(0.0, 0.0),
            glm.vec2(1.0, 0.0)
        ]

        self.positions = [
            v[0], v[1], v[2], v[3], # front
            v[0], v[3], v[4], v[5], # right
            v[0], v[5], v[6], v[1], # top
            v[1], v[6], v[7], v[2], # left
            v[7], v[4], v[3], v[2], # bottom
            v[4], v[7], v[6], v[5]  # back
        ]

        self.normals = [
            n[0], n[0], n[0], n[0], # front
            n[1], n[1], n[1], n[1], # right
            n[2], n[2], n[2], n[2], # top
            n[3], n[3], n[3], n[3], # left
            n[4], n[4], n[4], n[4], # bottom
            n[5], n[5], n[5], n[5]  # back
        ]

        self.tex_coords = [
            t[0], t[1], t[2], t[3], # front
            t[1], t[2], t[3], t[0], # right
            t[3], t[0], t[1], t[2], # top
            t[0], t[1], t[2], t[3], # left
            t[2], t[3], t[0], t[1], # bottom
            t[2], t[3], t[0], t[1]  # back
        ]

        self.indices = [
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

        vertices = core.to_vertex_array(self.positions, self.normals, self.tex_coords)
        vao = core.VAO(vertices, self.indices)
        super().__init__(vao, vertices, self.indices)

class Sphere(core.Mesh):
    def __init__(
        self,
        radius,
        stacks=16,
        sectors=16
    ):
        self.positions, self.normals, self.tex_coords, self.indices = self.get_vertices(radius, stacks, sectors)
        vertices = core.to_vertex_array(self.positions, self.normals, self.tex_coords)
        vao = core.VAO(vertices, self.indices)
        super().__init__(vao, vertices, self.indices)

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
# class Sphere(Primitives):
#         self.radius, self.subh, self.suba = radius, subh, suba
#         self.positions, self.indices, self.normals, self.tex_coords = self.get_vertices()
#         self.colors = get_color_by_pos(self.positions)
#         self.material = material

#         super().__init__(self.positions, self.colors, self.normals, self.tex_coords)

#         self.element_buff = glGenBuffers(1)
#         glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff)
#         glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

#     def get_vertices(self):
#         y, rst = [], []
#         cp, sp = [], []
#         for i in range(self.subh+1):
#             theta = i * np.pi / self.subh
#             y.append(self.radius * np.cos(theta))
#             rst.append(self.radius * np.sin(theta))
#         for i in range(self.suba+1):
#             phi = 2 * np.pi * i / self.suba
#             cp.append(np.cos(phi))
#             sp.append(np.sin(phi))

#         positions, normals, tex_coords = [], [], []
#         for i in range(self.subh):
#             for j in range(self.suba):
#                 vx0, vy0, vz0 = sp[j] * rst[i], y[i], cp[j] * rst[i]
#                 vx1, vy1, vz1 = sp[j] * rst[i+1], y[i+1], cp[j] * rst[i+1]
#                 vx2, vy2, vz2 = sp[j+1] * rst[i], y[i], cp[j+1] * rst[i]
#                 vx3, vy3, vz3 = sp[j+1] * rst[i+1], y[i+1], cp[j+1] * rst[i+1]

#                 if i < self.subh - 1:
#                     positions.append([vx0, vy0, vz0])
#                     positions.append([vx1, vy1, vz1])
#                     positions.append([vx3, vy3, vz3])

#                     normals.append([vx0 / self.radius, vy0 / self.radius, vz0 / self.radius])
#                     normals.append([vx1 / self.radius, vy1 / self.radius, vz1 / self.radius])
#                     normals.append([vx3 / self.radius, vy3 / self.radius, vz3 / self.radius])
                    
#                     u = np.arctan2(vx0 / self.radius, vz0 / self.radius) / (2 * np.pi) + 0.5
#                     v = vy0 / self.radius * 0.5 + 0.5
#                     tex_coords.append([u, v])

#                     u = np.arctan2(vx1 / self.radius, vz1 / self.radius) / (2 * np.pi) + 0.5
#                     v = vy1 / self.radius * 0.5 + 0.5
#                     tex_coords.append([u, v])

#                     u = np.arctan2(vx3 / self.radius, vz3 / self.radius) / (2 * np.pi) + 0.5
#                     v = vy3 / self.radius * 0.5 + 0.5
#                     tex_coords.append([u, v])

#                 if i > 0:
#                     positions.append([vx3, vy3, vz3])
#                     positions.append([vx2, vy2, vz2])
#                     positions.append([vx0, vy0, vz0])

#                     normals.append([vx3 / self.radius, vy3 / self.radius, vz3 / self.radius])
#                     normals.append([vx2 / self.radius, vy2 / self.radius, vz2 / self.radius])
#                     normals.append([vx0 / self.radius, vy0 / self.radius, vz0 / self.radius])

#                     u = np.arctan2(vx3 / self.radius, vz3 / self.radius) / (2 * np.pi) + 0.5
#                     v = vy3 / self.radius * 0.5 + 0.5
#                     tex_coords.append([u, v])

#                     u = np.arctan2(vx2 / self.radius, vz2 / self.radius) / (2 * np.pi) + 0.5
#                     v = vy2 / self.radius * 0.5 + 0.5
#                     tex_coords.append([u, v])

#                     u = np.arctan2(vx0 / self.radius, vz0 / self.radius) / (2 * np.pi) + 0.5
#                     v = vy0 / self.radius * 0.5 + 0.5
#                     tex_coords.append([u, v])

#         positions = np.array(positions, dtype=np.float32).flatten()
#         indices = np.array([i for i in range(len(positions) // 3)], dtype=np.uint32)
#         normals = np.array(normals, dtype=np.float32).flatten()
#         tex_coords = np.array(tex_coords, dtype=np.float32).flatten()
#         return positions, indices, normals, tex_coords

#     def render(self, shader: Shader, render_wireframe: bool=False):
#         if self.material != None:
#             shader.set_int("colorMode", 0)
#             self.material.update(shader)
#         else:
#             shader.set_int("colorMode", 1)

#         glBindVertexArray(self.vao)
#         glEnable(GL_POLYGON_OFFSET_FILL)
#         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
#         glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff)
#         glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

#         if render_wireframe == True:
#             glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
#             glLineWidth(2)
#             shader.set_int("colorMode", 2)
#             shader.set_vec4("uColor", glm.vec4(0))
#             glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

#         glDisable(GL_POLYGON_OFFSET_FILL)

# class Cylinder(Primitives):
#     def __init__(self,
#                  radius: float,
#                  height: float,
#                  n:      int,
#                  material=None):
#         self.radius, self.height, self.n = radius, height, n
#         self.positions, self.indices, self.normals, self.tex_coords = self.get_vertices()
#         self.colors = get_color_by_pos(self.positions)
#         self.material = material

#         super().__init__(self.positions, self.colors, self.normals, self.tex_coords)

#         # side - top - bottom
#         self.element_buff = glGenBuffers(3)
#         for i in range(3):
#             glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff[i])
#             glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices[i].nbytes, self.indices[i], GL_STATIC_DRAW)
        
#     def get_vertices(self):
#         top_positions, side_positions, bottom_positions = [], [], []
#         top_normals, side_normals, bottom_normals = [], [], []
#         top_indices, side_indices, bottom_indices = [], [], []
#         top_tex_coords, side_tex_coords, bottom_tex_coords = [], [], []
        
#         top_positions.append([0, self.height * 0.5, 0])
#         top_normals.append([0, 1, 0])
#         top_indices.append(0)
#         top_tex_coords.append([0.5, 0.5])

#         bottom_positions.append([0, -self.height * 0.5, 0])
#         bottom_normals.append([0, -1, 0])
#         bottom_indices.append(3 * self.n + 4)
#         bottom_tex_coords.append([0.5, 0.5])

#         for i in range(self.n+1):
#             theta = 2.0 * np.pi * i / self.n
#             x = self.radius * np.sin(theta)
#             z = self.radius * np.cos(theta)

#             top_positions.append([x, self.height/2, z])
#             top_normals.append([0, 1, 0])
#             top_indices.append(i+1)
#             top_tex_coords.append([x / self.radius * 0.5 + 0.5, z / self.radius * 0.5 + 0.5])
            
#             side_positions.append([x, self.height * 0.5, z])
#             side_positions.append([x, -self.height * 0.5, z])
#             side_normals.append([x / self.radius, 0, z / self.radius])
#             side_normals.append([x / self.radius, 0, z / self.radius])
#             side_indices.append(self.n + 2 + 2 * i)
#             side_indices.append(self.n + 3 + 2 * i)
#             side_tex_coords.append([i / self.n * np.pi / self.radius, self.height * self.radius * 0.5])
#             side_tex_coords.append([i / self.n * np.pi / self.radius, 0])

#             bottom_positions.append([x, -self.height * 0.5, z])
#             bottom_normals.append([0, -1, 0])
#             bottom_indices.append(3 * self.n + 5 + i)
#             bottom_tex_coords.append([x / self.radius * 0.5 + 0.5, z / self.radius * 0.5 + 0.5])

#         positions = np.concatenate((top_positions, side_positions, bottom_positions), dtype=np.float32).flatten()        
#         top_indices = np.array(top_indices, dtype=np.uint32)
#         side_indices = np.array(side_indices, dtype=np.uint32)
#         bottom_indices = np.array(bottom_indices, dtype=np.uint32)
#         normals = np.concatenate((top_normals, side_normals, bottom_normals), dtype=np.float32).flatten()
#         tex_coords = np.concatenate((top_tex_coords, side_tex_coords, bottom_tex_coords), dtype=np.float32).flatten()
#         return positions, (top_indices, side_indices, bottom_indices), normals, tex_coords

#     def render(self, shader: Shader, render_wireframe: bool=False):
#         if self.material != None:
#             shader.set_int("colorMode", 0)
#             self.material.update(shader)
#         else:
#             shader.set_int("colorMode", 1)

#         glBindVertexArray(self.vao)

#         glEnable(GL_POLYGON_OFFSET_FILL)
#         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

#         drawing_mode = [ GL_TRIANGLE_FAN, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN ]
#         for i in range(3):
#             glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff[i])
#             glDrawElements(drawing_mode[i], self.indices[i].nbytes, GL_UNSIGNED_INT, None)
        
#         if render_wireframe:
#             glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
#             glLineWidth(2)
#             shader.set_int("colorMode", 2)
#             shader.set_vec4("uColor", glm.vec4(0))
#             for i in range(3):
#                 glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff[i])
#                 glDrawElements(drawing_mode[i], self.indices[i].nbytes, GL_UNSIGNED_INT, None)

#         glDisable(GL_POLYGON_OFFSET_FILL)

# class Cone(Primitives):
#     def __init__(self,
#                  radius: float,
#                  height: float,
#                  n:      int,
#                  material=None):
#         self.radius, self.height, self.n = radius, height, n
#         self.positions, self.indices, self.normals, self.tex_coords = self.get_vertices()
#         self.colors = get_color_by_pos(self.positions)
#         self.material = material

#         super().__init__(self.positions, self.colors, self.normals, self.tex_coords)

#         self.element_buff = glGenBuffers(2)
#         for i in range(2):
#             glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff[i])
#             glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices[i].nbytes, self.indices[i], GL_STATIC_DRAW)
    
#     def get_vertices(self):
#         positions = []
#         side_idx, bottom_idx = [], []
#         normals = []
#         # TODO: texture coordinates for bottom side
#         tex_coords = []

#         half_height = self.height * 0.5

#         # side 
#         positions.append([0, half_height, 0])
#         normals.append([0, 1, 0])
#         side_idx.append(0)
#         tex_coords.append([0.5, 0.5])

#         for i in range(self.n+1):
#             theta = 2.0 * np.pi * i / self.n
#             x = self.radius * np.sin(theta)
#             z = self.radius * np.cos(theta)

#             positions.append([x, -half_height, z])
#             normals.append([x / self.radius, 0, z / self.radius])
#             side_idx.append(i+1)
#             tex_coords.append([x / self.radius * 0.5 + 0.5, z / self.radius * 0.5 + 0.5])

#         # bottom
#         positions.append([0, -half_height, 0])
#         normals.append([0, -1, 0])
#         bottom_idx.append(self.n+2)
#         tex_coords.append([0.5, 0.5])

#         for i in range(self.n+1):
#             theta = 2.0 * np.pi * i / self.n
#             x = self.radius * np.sin(theta)
#             z = self.radius * np.cos(theta)

#             positions.append([x, -half_height, z])
#             normals.append([0, -1, 0])
#             bottom_idx.append(self.n + 3 + i)
#             tex_coords.append([x / self.radius * 0.5 + 0.5, z / self.radius * 0.5 + 0.5])
        
#         positions = np.array(positions, dtype=np.float32).flatten()
#         side_idx = np.array(side_idx, dtype=np.uint32)
#         bottom_idx = np.array(bottom_idx, dtype=np.uint32)
#         normals = np.array(normals, dtype=np.float32).flatten()
#         tex_coords = np.array(tex_coords, dtype=np.float32).flatten()
#         return positions, (side_idx, bottom_idx), normals, tex_coords
    
#     def render(self, shader: Shader, render_wireframe: bool=False):
#         if self.material != None:
#             shader.set_int("colorMode", 0)
#             self.material.update(shader)
#         else:
#             shader.set_int("colorMode", 1)

#         glBindVertexArray(self.vao)

#         glEnable(GL_POLYGON_OFFSET_FILL)
#         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

#         for i in range(2):
#             glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff[i])
#             glDrawElements(GL_TRIANGLE_FAN, self.indices[i].nbytes, GL_UNSIGNED_INT, None)
        
#         if render_wireframe:
#             glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
#             glLineWidth(2)
#             shader.set_int("colorMode", 2)
#             shader.set_vec4("uColor", glm.vec4(0))
#             for i in range(2):
#                 glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff[i])
#                 glDrawElements(GL_TRIANGLE_FAN, self.indices[i].nbytes, GL_UNSIGNED_INT, None)

#         glDisable(GL_POLYGON_OFFSET_FILL)

# class Ground(Primitives):
#     def __init__(self,
#                 width: float,
#                 height: float,
#                 texture_distance: float=10,
#                 material=None):
#         self.width, self.height, self.texture_distance = width, height, texture_distance
#         self.positions, self.indices, self.normals, self.tex_coords = self.get_vertices()
#         self.colors = get_color(self.positions)
#         self.material = material

#         self.element_buff = glGenBuffers(1)
#         glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff)
#         glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

#         super().__init__(self.positions, self.colors, self.normals, self.tex_coords)
    
#     def get_vertices(self):
#         positions = [
#             [-self.width * 0.5, 0, -self.height * 0.5],
#             [-self.width * 0.5, 0, self.height * 0.5],
#             [self.width * 0.5, 0, -self.height * 0.5],
#             [self.width * 0.5, 0, self.height * 0.5],
#             [-self.width * 0.5, 0, -self.height * 0.5],
#             [-self.width * 0.5, 0, self.height * 0.5],
#             [self.width * 0.5, 0, -self.height * 0.5],
#             [self.width * 0.5, 0, self.height * 0.5],
#         ]
#         indices = [[0, 1, 2, 2, 1, 3] * 2]
#         normals = [[0, 1, 0] * 4, [0, -1, 0] * 4]
#         tex_coords = [
#             [0, 0],
#             [0, self.height / self.texture_distance],
#             [self.width/ self.texture_distance, 0],
#             [self.width / self.texture_distance, self.height / self.texture_distance],
#             [0, 0],
#             [0, self.height / self.texture_distance],
#             [self.width/ self.texture_distance, 0],
#             [self.width / self.texture_distance, self.height / self.texture_distance]
#         ]

#         positions = np.array(positions, dtype=np.float32).flatten()
#         indices = np.array(indices, dtype=np.uint32).flatten()
#         normals = np.array(normals, dtype=np.float32).flatten()
#         tex_coords = np.array(tex_coords, dtype=np.float32).flatten()
#         return positions, indices, normals, tex_coords
    
#     def render(self, shader):
#         if self.material != None:
#             shader.set_int("colorMode", 0)
#             self.material.update(shader)
#         else:
#             shader.set_int("colorMode", 1)

#         glBindVertexArray(self.vao)
        
#         glEnable(GL_POLYGON_OFFSET_FILL)
#         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

#         glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff)
#         glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

#         glDisable(GL_POLYGON_OFFSET_FILL)

# class Arrow(Primitives):
#     def __init__(self,
#                  length: float,
#                  radius: float,
#                  material=None):
#         self.length, self.radius = length, radius
#         self.material = material
        
#         self.body = Cylinder(radius, length * 0.8, 16, material)
#         self.head = Cone(radius * 2, length * 0.2, 16, material)
    
#     def render(self, shader: Shader, R: glm.mat4=glm.mat4(1.0)):
#         M = glm.translate(glm.mat4(1.0), glm.vec3(0, self.length * 0.4, 0))
#         shader.set_mat4("M", R * M)
#         self.body.render(shader, False)
#         M = glm.translate(glm.mat4(1.0), glm.vec3(0, self.length * 0.9, 0))
#         shader.set_mat4("M", R * M)
#         self.head.render(shader, False)