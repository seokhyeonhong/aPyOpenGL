from __future__ import annotations
try:
    import fbx
    from . import fbxparser
except ImportError:
    print("Warning: Failed to import fbx. Please install fbx sdk and rebuild aPyOpenGL.")

import os
import pickle
from tqdm import tqdm

from . import core
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
    "TransparentColor":  TextureType.eDIFFUSE,
}

def _get_resampled_scene(scene_and_frame_idx):
    scene, frame_idx = scene_and_frame_idx
    return fbxparser.keyframe.resample(scene, frame_idx)

def _parse_motion(scene_and_frame_idx, names):
    scene, frame_idx = scene_and_frame_idx
    
    rotations = fbxparser.keyframe.get_rotations_from_resampled(names, scene, len(frame_idx))
    positions = fbxparser.keyframe.get_positions_from_resampled(names[0], scene, len(frame_idx))

    return rotations, positions

class Parser:
    def __init__(self, path, scale, save):
        self.path = path
        self.scale = scale
        self.save = save
        self.parser = fbxparser.FBXParser(path)
        self._init_character_data(scale)
        self._init_mesh_data(scale)
    
    def pkl_path(self, pkl_type):
        dir_path = os.path.dirname(self.path)
        filename = os.path.basename(self.path).split(".")[0]
        save_dir = os.path.join(dir_path, "aPyOpenGL-pkl")
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, f"{filename}_{pkl_type}.pkl")
    
    def _init_character_data(self, scale):
        self.char_data = fbxparser.CharacterData()
        root = self.parser.scene.GetRootNode()
        self.char_data.name = root.GetName()
        for i in range(root.GetChildCount()):
            fbxparser.parse_nodes_by_type(root.GetChild(i), self.char_data.joint_data, -1, fbx.FbxNodeAttribute.eSkeleton, scale)

    def _init_mesh_data(self, scale):
        # return pickled mesh data if exists and is newer than fbx file
        mesh_pkl_path = self.pkl_path("mesh")
        if os.path.exists(mesh_pkl_path) and os.path.getmtime(mesh_pkl_path) > os.path.getmtime(self.path):
            with open(mesh_pkl_path, "rb") as f:
                self.mesh_data = pickle.load(f)
            return

        mesh_nodes = []

        root = self.parser.scene.GetRootNode()
        self._load_mesh_recursive(root, mesh_nodes)
        
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
        
        # save mesh data only when mesh_data exists
        if len(self.mesh_data) > 0:
            with open(mesh_pkl_path, "wb") as f:
                pickle.dump(self.mesh_data, f, pickle.HIGHEST_PROTOCOL)
    
    def _load_mesh_recursive(self, node, mesh_nodes):
        for i in range(node.GetNodeAttributeCount()):
            attrib = node.GetNodeAttributeByIndex(i)
            if attrib.GetAttributeType() == fbx.FbxNodeAttribute.eMesh:
                mesh_nodes.append(node)
        
        for i in range(node.GetChildCount()):
            self._load_mesh_recursive(node.GetChild(i), mesh_nodes)
    
    def motions(self, joints: list[fbxparser.JointData]):
        # return pickled motion if exists and is newer than fbx file
        motion_pkl_path = self.pkl_path("motion")
        if os.path.exists(motion_pkl_path) and os.path.getmtime(motion_pkl_path) > os.path.getmtime(self.path):
            with open(motion_pkl_path, "rb") as f:
                return pickle.load(f)
        
        # create skeleton
        skeleton = Skeleton()
        for joint in joints:
            skeleton.add_joint(joint.name, pre_quat=joint.pre_quat, local_pos=joint.local_T, parent_idx=joint.parent_idx)

        # get keyframes
        scenes = self.parser.get_scene_keyframes(self.scale)
        names = [joint.name for joint in joints]

        # resample
        frame_set = []
        for scene in tqdm(scenes, desc="Resampling"):
            frame_idx = [i for i in range(scene.start_frame, scene.end_frame + 1)]
            frame_set.append(frame_idx)
        
        # resampled_scenes = util.run_parallel_sync(_get_resampled_scene, zip(scenes, frame_set), desc="Resampling scenes")
        resampled_scenes = []
        for i in tqdm(range(len(scenes)), desc="Resampling scenes"):
            resampled_scenes.append(_get_resampled_scene((scenes[i], frame_set[i])))

        # parse
        # rotations_and_positions = util.run_parallel_sync(_parse_motion, zip(resampled_scenes, frame_set), names=names, desc="Parsing motions")
        rotations_and_positions = []
        for i in tqdm(range(len(resampled_scenes)), desc="Parsing motions"):
            rotations_and_positions.append(_parse_motion((resampled_scenes[i], frame_set[i]), names=names))

        # create motion
        motion_set = []
        for rot, pos in rotations_and_positions:
            poses = []
            for i in range(len(rot)):
                poses.append(Pose(skeleton, local_quats=rot[i], root_pos=pos[i]))
            
            motion = Motion(poses, fps=self.parser.get_scene_fps(), name=self.parser.filepath)
            motion_set.append(motion)
        
        if len(motion_set) > 0:
            with open(motion_pkl_path, "wb") as f:
                pickle.dump(motion_set, f, pickle.HIGHEST_PROTOCOL)
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
                    tangents=data.tangents,
                    bitangents=data.bitangents,
                    lbs_indices1=data.skinning_data.joint_indices1,
                    lbs_weights1=data.skinning_data.joint_weights1,
                    lbs_indices2=data.skinning_data.joint_indices2,
                    lbs_weights2=data.skinning_data.joint_weights2
                )
                mesh.joint_names = data.skinning_data.joint_names
                mesh.name_to_idx = data.skinning_data.name_to_idx
                mesh.bind_xform_inv = data.skinning_data.offset_xform
                mesh.control_point_idx_to_vertex_idx = data.control_point_idx_to_vertex_idx
                mesh.vertex_idx_to_control_point_idx = data.vertex_idx_to_control_point_idx
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
                    gl_texture_type = FBX_PROPERTY_NAMES.get(str(texture_info.property), "unknown")
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
            skeleton.add_joint(joint.name, pre_quat=joint.pre_quat, local_pos=joint.local_T, parent_idx=joint.parent_idx)

        return skeleton

    def model(self) -> Model:
        meshes   = self.meshes_and_materials()
        skeleton = self.skeleton()

        meshes   = meshes if len(meshes) > 0 else None
        skeleton = skeleton if skeleton.num_joints > 0 else None

        return Model(meshes=meshes, skeleton=skeleton)
    
    def motions(self) -> list[Motion]:
        return self.parser.motions(self.parser.char_data.joint_data)
    
    def fps(self):
        return self.parser.parser.get_scene_fps()