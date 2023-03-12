import os
import fbx
import pickle

from pymovis.motion.data import fbx_mesh, fbx_texture, fbx_material, fbx_skeleton, fbx_parser, fbx_skin
from pymovis.motion.core import Skeleton

from pymovis.vis.core import MeshGL, VertexGL, VAO
from pymovis.vis.material import Material
from pymovis.vis.model import Model
from pymovis.vis.texture import TextureType, TextureLoader

FBX_PROPERTY_NAMES = {
    "DiffuseColor":      TextureType.eDIFFUSE,
    "EmissiveColor":     TextureType.eEMISSIVE,
    "SpecularColor":     TextureType.eSPECULAR,
    "ShininessExponent": TextureType.eGLOSSINESS,
    "NormalMap":         TextureType.eNORMAL,
}

def find_texture_type(type_name):
    find = FBX_PROPERTY_NAMES.get(type_name)
    return find if find is not None else TextureType.eUNKNOWN

class Parser:
    def __init__(self, path, scale, save):
        self.path = path
        self.scale = scale
        self.save = save
        self.parser = fbx_parser.FBXParser(path)
        self.init_character_data(scale)
        self.init_mesh_data(scale)

    def init_character_data(self, scale):
        filename = os.path.splitext(os.path.basename(self.path))[0]
        save_path = os.path.join(os.path.dirname(self.path), f"{filename}_character_data.pkl")
        # if os.path.exists(save_path) and self.save:
        #     with open(save_path, "rb") as f:
        #         self.char_data = pickle.load(f)
        #     return

        self.char_data = fbx_parser.CharacterData()
        root = self.parser.scene.GetRootNode()
        is_root_found = False
        for i in range(root.GetNodeAttributeCount()):
            attr = root.GetNodeAttributeByIndex(i)
            if attr.GetATtributeType() == fbx.FbxNodeAttribute.eSkeleton:
                is_root_found = True
        
        if not is_root_found:
            root = root.GetChild(0)

        self.char_data.name = root.GetName()
        for i in range(root.GetChildCount()):
            fbx_skeleton.parse_nodes_by_type(root.GetChild(i), self.char_data.joint_data, -1, fbx.FbxNodeAttribute.eSkeleton, scale)
        
        # if not os.path.exists(save_path) and self.save:
        #     with open(save_path, "wb") as f:
        #         pickle.dump(self.char_data, f, pickle.HIGHEST_PROTOCOL)

    def init_mesh_data(self, scale):
        filename = os.path.splitext(os.path.basename(self.path))[0]
        save_path = os.path.join(os.path.dirname(self.path), f"{filename}_mesh_data.pkl")
        # if os.path.exists(save_path) and self.save:
        #     with open(save_path, "rb") as f:
        #         self.mesh_data = pickle.load(f)
        #     return

        mesh_nodes = []

        root = self.parser.scene.GetRootNode()
        self.__load_mesh_recursive(root, mesh_nodes)
        
        self.mesh_data = []
        for i in range(len(mesh_nodes)):
            node = mesh_nodes[i]
            fbx_mesh_ = node.GetMesh()

            # read materials and textures
            textures = fbx_texture.get_textures(fbx_mesh_)
            materials = fbx_material.get_materials(fbx_mesh_)
            material_connection = fbx_material.get_polygon_material_connection(fbx_mesh_)

            for i in range(len(textures)):
                texture_set = False
                for j in range(len(materials)):
                    if textures[i].connected_material == materials[j].material_id:
                        materials[j].texture_ids.append(i)
                        texture_set = True
                        break
                
                if not texture_set:
                    print(f"Texture is NOT used {textures[i].filename}")
                textures[i].is_used = texture_set

            mesh_data = fbx_mesh.get_mesh_data(fbx_mesh_, scale)
            mesh_data.textures = textures
            mesh_data.materials = materials
            mesh_data.polygon_material_connection = material_connection

            mesh_data.is_skinned = fbx_skin.get_skinning(
                mesh_data.skinning_data,
                fbx_mesh_,
                mesh_data.control_point_idx_to_vertex_idx,
                len(mesh_data.positions),
                scale)
            
            self.mesh_data.append(mesh_data)
        
        # if not os.path.exists(save_path) and self.save:
        #     with open(save_path, "wb") as f:
        #         pickle.dump(self.mesh_data, f, pickle.HIGHEST_PROTOCOL)
    
    def __load_mesh_recursive(self, node, mesh_nodes):
        for i in range(node.GetNodeAttributeCount()):
            attrib = node.GetNodeAttributeByIndex(i)
            if attrib.GetAttributeType() == fbx.FbxNodeAttribute.eMesh:
                mesh_nodes.append(node)
        
        for i in range(node.GetChildCount()):
            self.__load_mesh_recursive(node.GetChild(i), mesh_nodes)

class FBX:
    def __init__(self, filename, scale=0.01, save=True):
        self.parser = Parser(filename, scale, save)
        self.scale = scale

    def meshes_and_materials(self):
        mesh_data = self.parser.mesh_data

        results = []
        for data in mesh_data:
            mesh = MeshGL()

            if data.is_skinned:
                mesh.is_skinned = True
                mesh.vertices = VertexGL.make_vertex_array(data.positions, data.normals, data.uvs, data.skinning_data.joint_indices1, data.skinning_data.joint_weights1, data.skinning_data.joint_indices2, data.skinning_data.joint_weights2)
                mesh.joint_order = data.skinning_data.joint_order
                mesh.name_to_idx = data.skinning_data.name_to_idx
                mesh.joint_bind_trf_inv = data.skinning_data.offset_transform
            else:
                mesh.is_skinned = False
                mesh.vertices = VertexGL.make_vertex_array(data.positions, data.normals, data.uvs)
            
            # set materials and texture
            id_to_material_idx = {}
            gl_materials = []
            for i in range(len(data.materials)):
                material_info = data.materials[i]
                id_to_material_idx[material_info.material_id] = i

                gl_material = Material()
                gl_material.albedo = material_info.diffuse

                for j in range(len(material_info.texture_ids)):
                    texture_id = material_info.texture_ids[j]
                    texture_info = data.textures[texture_id]

                    gl_texture = TextureLoader.load(texture_info.filename)
                    gl_texture_type = find_texture_type(texture_info.property)
                    gl_material.set_texture(gl_texture_type, gl_texture)
                
                gl_materials.append(gl_material)
            
            # set vertex material connection
            for i in range(len(data.polygon_material_connection)):
                material_id = data.polygon_material_connection[i]
                material_id = id_to_material_idx[material_id]

                idx0 = data.indices[i * 3 + 0]
                idx1 = data.indices[i * 3 + 1]
                idx2 = data.indices[i * 3 + 2]

                mesh.vertices[idx0].material_id = material_id
                mesh.vertices[idx1].material_id = material_id
                mesh.vertices[idx2].material_id = material_id
            
            mesh.indices = data.indices
            mesh.vao = VAO.from_vertex_array(mesh.vertices, mesh.indices)

            results.append((mesh, gl_materials))
    
        return results
    
    def skeleton(self):
        skeleton = Skeleton()
        char_data = self.parser.char_data
        joints = char_data.joint_data
        for joint in joints:
            # pre_R = npmotion.Q_to_R(np.array(joint.pre_R)).squeeze()
            skeleton.add_joint(joint.name, joint.parent_idx)#, pre_R)
        return skeleton

    def model(self):
        gl_meshes = self.meshes_and_materials()
        skeleton  = self.skeleton()

        mesh_exist     = (len(gl_meshes) > 0)
        skeleton_exist = (skeleton.num_joints > 0)

        if mesh_exist and skeleton_exist:
            return Model(gl_meshes=gl_meshes, skeleton=skeleton)
        elif mesh_exist:
            return Model(gl_meshes=gl_meshes)
        elif skeleton_exist:
            return Model(skeleton=skeleton)
        else:
            raise Exception("No mesh or skeleton data")