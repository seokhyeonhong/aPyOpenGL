from __future__ import annotations

import os
from OpenGL.GL import *
import glm

from .core import VAO, VertexGL
from .material import Material

def parse_obj(path, scale):
    if not path.endswith(".obj"):
        raise Exception(f"File must be .obj format, but got {path}")
    
    positions, uvs, normals, faces = [], [], [], []
    mtl_files = []
    current_mtl = None

    with open(path, "r") as f:
        lines = f.readlines()

        for line_idx, line in enumerate(lines):
            tokens = line.strip().split()
            if not tokens:
                continue
            
            prefix = tokens[0]

            # material
            if prefix == "mtllib":
                file_path = os.path.join(os.path.dirname(path), tokens[1])
                mtl_files.append(file_path)

            # group
            elif prefix == "g":
                continue

            # vertex
            elif prefix == "v":
                vertex = glm.vec3(list(map(float, tokens[1:]))) * scale
                positions.append(vertex)
            
            # uv
            elif prefix == "vt":
                tex_coord = glm.vec2(list(map(float, tokens[1:])))
                uvs.append(tex_coord)
            
            # normal
            elif prefix == "vn":
                normal = glm.vec3(list(map(float, tokens[1:])))
                normals.append(normal)
            
            # face
            elif prefix == "f":
                if len(tokens[1:]) == 3:
                    # exception for no material
                    if current_mtl is None:
                        raise Exception("Face must be preceded by a material")
                    
                    # parse vertices
                    vertices = [tokens[1].split("/"), tokens[2].split("/"), tokens[3].split("/")]
                    for vtn in vertices:
                        position_index = int(vtn[0]) - 1
                        uv_index       = int(vtn[1]) - 1 if len(vtn) > 1 and vtn[1] != "" else -1
                        normal_index   = int(vtn[2]) - 1 if len(vtn) > 2 and vtn[2] != "" else -1
                        faces.append((position_index, uv_index, normal_index, current_mtl))
                else:
                    raise Exception(f"We only support faces with three vertices, but got {len(tokens[1:])} vertices: {line}")
            
            # material
            elif prefix == "usemtl":
                current_mtl = tokens[1]
            
            # unknown
            else:
                print("Line {}: Unknown line {}".format(line_idx, line.replace("\n", "")))
                continue

    return positions, uvs, normals, faces, mtl_files

def parse_mtl(path):
    if not path.endswith(".mtl"):
        raise Exception(f"File must be .mtl format, but got {path}")
    
    materials = {}
    material_name = None

    with open(path, "r") as f:
        lines = f.readlines()

        for line_idx, line in enumerate(lines):
            tokens = line.strip().split()
            if not tokens:
                continue

            prefix = tokens[0]

            # newmtl
            if prefix == "newmtl":
                material_name = tokens[1]
                materials[material_name] = Material()
            
            # ambient - not used in our renderer
            elif prefix == "Ka":
                # ambient = glm.vec3(list(map(float, tokens[1:])))
                pass
            
            # diffuse - albedo in our renderer
            # This is because "diffuse" in our renderer is the color of diffuse light
            elif prefix == "Kd":
                materials[material_name].albedo = glm.vec3(list(map(float, tokens[1:])))
            
            # specular
            elif prefix == "Ks":
                materials[material_name].specular = glm.vec3(list(map(float, tokens[1:])))
            
            # shininess
            elif prefix == "Ns":
                materials[material_name].shininess = float(tokens[1])
            
            # # texture - not supported yet
            # elif prefix == "map_Kd":
            #     materials[material_name]["texture"] = tokens[1]

            else:
                print("Line {}: Unknown line {}".format(line_idx, line.replace("\n", "")))
                continue
    
    return materials

def make_vertex(face, positions, uvs, normals, name_to_mtl_idx):
    p_idx, uv_idx, n_idx, mtl_name = face

    vertex = VertexGL()
    vertex.position    = positions[p_idx]
    vertex.uv          = uvs[uv_idx] if uv_idx != -1 else glm.vec2(0.0)
    vertex.normal      = normals[n_idx] if n_idx != -1 else glm.vec3(0.0)
    vertex.material_id = name_to_mtl_idx[mtl_name]

    return vertex

class Obj(VAO):
    def __init__(self, path, scale=1.0):
        v_array, v_index, materials = Obj.generate_vertices(path, scale)
        vao = VAO.from_vertex_array(v_array, v_index)
        super().__init__(vao.id, vao.vbos, vao.ebo, vao.indices)

        self.materials = materials
    
    @staticmethod
    def generate_vertices(path, scale=1.0):
        positions, uvs, normals, faces, mtl_files = parse_obj(path, scale)

        # parse materials
        mtl_dict = {}
        for file in mtl_files:
            mtl_dict.update(parse_mtl(file))
        
        materials = [mtl_dict[name] for name in mtl_dict.keys()]
        name_to_mtl_idx = {name: idx for idx, name in enumerate(mtl_dict.keys())}

        # generate vertices
        vertex_array = [VertexGL(
            position    = positions[p_idx],
            normal      = normals[n_idx] if n_idx != -1 else glm.vec3(0),
            uv          = uvs[uv_idx] if uv_idx != -1 else glm.vec2(0),
            material_id = name_to_mtl_idx[mtl_name]
        ) for p_idx, uv_idx, n_idx, mtl_name in faces]

        vertex_index = list(range(len(vertex_array)))

        return vertex_array, vertex_index, materials