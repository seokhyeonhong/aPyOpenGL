from __future__ import annotations
import fbx
import numpy as np
import os
import pickle

from . import core, fbxparser, keyframe

from .motion   import Skeleton, Pose, Motion
from .material import Material
from .model    import Model
from .texture  import TextureType, TextureLoader

FBX_PROPERTY_NAMES = {
    "DiffuseColor":      TextureType.eDIFFUSE,
    "EmissiveColor":     TextureType.eEMISSIVE,
    "SpecularColor":     TextureType.eSPECULAR,
    "ShininessExponent": TextureType.eGLOSSINESS,
    "NormalMap":         TextureType.eNORMAL,
}

def find_texture_type(type_name):
    find = FBX_PROPERTY_NAMES.get(str(type_name))
    return find if find is not None else "unknown"

class Parser:
    def __init__(self, path, scale, save):
        self.path = path
        self.scale = scale
        self.save = save
        self.parser = fbxparser.FBXParser(path)
        self.init_character_data(scale)
        self.init_mesh_data(scale)

    def init_character_data(self, scale):
        self.char_data = fbxparser.CharacterData()
        root = self.parser.scene.GetRootNode()
        self.char_data.name = root.GetName()
        for i in range(root.GetChildCount()):
            fbxparser.parse_nodes_by_type(root.GetChild(i), self.char_data.joint_data, -1, fbx.FbxNodeAttribute.eSkeleton, scale)

    def init_mesh_data(self, scale):
        mesh_nodes = []

        root = self.parser.scene.GetRootNode()
        self.__load_mesh_recursive(root, mesh_nodes)
        
        self.mesh_data = []
        for i in range(len(mesh_nodes)):
            node = mesh_nodes[i]
            fbx_mesh_ = node.GetMesh()

            # read materials and textures
            textures = fbxparser.get_textures(fbx_mesh_)
            materials = fbxparser.get_materials(fbx_mesh_)
            material_connection = fbxparser.get_polygon_material_connection(fbx_mesh_)

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

            mesh_data = fbxparser.get_mesh_data(fbx_mesh_, scale)
            mesh_data.textures = textures
            mesh_data.materials = materials
            mesh_data.polygon_material_connection = material_connection

            mesh_data.is_skinned = fbxparser.get_skinning(
                mesh_data.skinning_data,
                fbx_mesh_,
                mesh_data.control_point_idx_to_vertex_idx,
                len(mesh_data.positions),
                scale
            )
            
            self.mesh_data.append(mesh_data)
    
    def __load_mesh_recursive(self, node, mesh_nodes):
        for i in range(node.GetNodeAttributeCount()):
            attrib = node.GetNodeAttributeByIndex(i)
            if attrib.GetAttributeType() == fbx.FbxNodeAttribute.eMesh:
                mesh_nodes.append(node)
        
        for i in range(node.GetChildCount()):
            self.__load_mesh_recursive(node.GetChild(i), mesh_nodes)
    
    def motions(self, joints: list[fbxparser.JointData]):
        skeleton = Skeleton()
        for joint in joints:
            skeleton.add_joint(joint.name, pre_Q=joint.pre_Q, local_p=joint.local_T, parent_idx=joint.parent_idx)

        scenes = self.parser.get_scene_keyframes(self.scale)
        names = [joint.name for joint in joints]
        resampled_scenes = []

        frame_set = []
        for scene in scenes:
            frame_idx = [i for i in range(scene.start_frame, scene.end_frame + 1)]
            frame_set.append(frame_idx)
            resampled_scenes.append(keyframe.resample(scene, frame_idx))

        motion_set = []
        root_name = skeleton.joints[0].name
        for i in range(len(resampled_scenes)):
            scene = resampled_scenes[i]
            nof = len(frame_set[i])
            
            rotations = keyframe.get_rotations_from_resampled(names, scene, nof)
            positions = keyframe.get_positions_from_resampled(root_name, scene, nof)

            poses = []
            for i in range(nof):
                poses.append(Pose(skeleton, local_quats=rotations[i], root_pos=positions[i]))

            motion_set.append(Motion(poses, fps=self.parser.get_scene_fps(), name=self.parser.filepath))
        
        return motion_set

class FBX:
    def __init__(self, filename, scale=0.01, save=True):
        self.filename = os.path.basename(filename).split(".")[0]
        self.parser = Parser(filename, scale, save)
        self.scale = scale

    def meshes_and_materials(self) -> list[tuple[core.MeshGL, Material]]:
        mesh_data = self.parser.mesh_data

        results = []
        for data in mesh_data:
            mesh = core.MeshGL()

            if data.is_skinned:
                mesh.is_skinned = True
                mesh.vertices = core.to_vertex_array(
                    data.positions,
                    data.normals,
                    data.uvs,
                    data.tangents,
                    data.bitangents,
                    data.skinning_data.joint_indices1,
                    data.skinning_data.joint_weights1,
                    data.skinning_data.joint_indices2,
                    data.skinning_data.joint_weights2
                )
                mesh.joint_names = data.skinning_data.joint_names
                mesh.name_to_idx = data.skinning_data.name_to_idx
                mesh.joint_bind_xform_inv = data.skinning_data.offset_xform
            else:
                mesh.is_skinned = False
                mesh.vertices = core.to_vertex_array(data.positions, data.normals, data.uvs, data.tangents, data.bitangents)
            
            # set materials and texture
            id_to_material_idx = {}
            materials = []
            for i in range(len(data.materials)):
                material_info = data.materials[i]
                id_to_material_idx[material_info.material_id] = i

                material = Material()
                material.albedo = material_info.diffuse

                for j in range(len(material_info.texture_ids)):
                    texture_id = material_info.texture_ids[j]
                    texture_info = data.textures[texture_id]

                    gl_texture = TextureLoader.load(texture_info.filename)
                    gl_texture_type = find_texture_type(texture_info.property)
                    material.set_texture(gl_texture, gl_texture_type)
                
                materials.append(material)
            
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
            mesh.vao = core.bind_mesh(mesh.vertices, mesh.indices, compute_tangent=False)

            results.append((mesh, materials))
    
        return results
    
    def skeleton(self) -> Skeleton:
        skeleton = Skeleton()
        char_data = self.parser.char_data
        joints = char_data.joint_data
        for joint in joints:
            skeleton.add_joint(joint.name, pre_Q=joint.pre_Q, local_p=joint.local_T, parent_idx=joint.parent_idx)
        return skeleton

    def model(self):
        meshes   = self.meshes_and_materials()
        skeleton = self.skeleton()

        meshes   = meshes if len(meshes) > 0 else None
        skeleton = skeleton if skeleton.num_joints() > 0 else None

        return Model(meshes=meshes, skeleton=skeleton)
    
    def motions(self) -> list[Motion]:
        return self.parser.motions(self.parser.char_data.joint_data)