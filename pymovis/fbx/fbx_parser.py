import fbx
import glm

from pymovis.fbx import fbx_animation

FbxNode = fbx.FbxNode
FbxAnimStack = fbx.FbxAnimStack
FbxAnimLayer = fbx.FbxAnimLayer
FbxCriteria = fbx.FbxCriteria

class FBXParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.init_scene()
    
    def init_scene(self):
        # initialize SDK manager that handles memory management
        self.manager = fbx.FbxManager.Create()
        ios = fbx.FbxIOSettings.Create(self.manager, fbx.IOSROOT)
        self.manager.SetIOSettings(ios)

        # create an importer
        importer = fbx.FbxImporter.Create(self.manager, "")
        success = importer.Initialize(self.filepath, -1, self.manager.GetIOSettings())
        if not success:
            raise Exception("Failed to initialize importer")
        
        # create a new scene so that it can be populated by the imported file
        self.scene = fbx.FbxScene.Create(self.manager, "scene")

        # import the contents of the file into the scene
        importer.Import(self.scene)

        # time setting to 30 fps
        time_settings = self.scene.GetGlobalSettings()
        time_mode = time_settings.GetTimeMode()
        time_settings.SetTimeMode(fbx.FbxTime.eFrames30)

        # triangulate
        fbx.FbxGeometryConverter(self.manager).Triangulate(self.scene, True)

        # bake
        self.scene.ConnectSrcObject(self.scene)
        self.bake_node(self.scene.GetRootNode())

        # convert pivot
        self.scene.GetRootNode().ConvertPivotAnimationRecursive(None, FbxNode.eDestinationPivot, 30, True)

        # name check
        name_counter = {}
        self.check_same_name(self.scene.GetRootNode(), name_counter)

        # the file is imported, so get rid of the importer
        importer.Destroy()

    def bake_node(self, node):
        zero = fbx.FbxVector4(0, 0, 0)

        # pivot converting
        node.SetPivotState(FbxNode.eSourcePivot, FbxNode.ePivotActive)
        node.SetPivotState(FbxNode.eDestinationPivot, FbxNode.ePivotActive)

        # set all these to 0 and bake them
        node.SetPostRotation(FbxNode.eDestinationPivot, zero)
        node.SetPreRotation(FbxNode.eDestinationPivot, node.GetPreRotation(FbxNode.eSourcePivot))
        node.SetRotationOffset(FbxNode.eDestinationPivot, zero)
        node.SetScalingOffset(FbxNode.eDestinationPivot, zero)
        node.SetRotationPivot(FbxNode.eDestinationPivot, zero)
        node.SetScalingPivot(FbxNode.eDestinationPivot, zero)

        # import in a system that supports rotation order
        # if the rotation order is not supported, do this instead:
        # node.SetRotationOrder(FbxNode.eDestinationPivot, FbxNode.eEulerXYZ)
        order = node.GetRotationOrder(FbxNode.eSourcePivot)
        node.SetRotationOrder(FbxNode.eDestinationPivot, order)

        # geometric transforms
        # if not supported, set them to zero
        node.SetGeometricTranslation(FbxNode.eDestinationPivot, node.GetGeometricTranslation(FbxNode.eSourcePivot))
        node.SetGeometricRotation(FbxNode.eDestinationPivot, node.GetGeometricRotation(FbxNode.eSourcePivot))
        node.SetGeometricScaling(FbxNode.eDestinationPivot, node.GetGeometricScaling(FbxNode.eSourcePivot))

        # idem for quaternions
        node.SetQuaternionInterpolation(FbxNode.eDestinationPivot, node.GetQuaternionInterpolation(FbxNode.eSourcePivot))

        node.ConvertPivotAnimationRecursive(None, FbxNode.eDestinationPivot, 30.0, True)

        for i in range(node.GetChildCount()):
            self.bake_node(node.GetChild(i))

    def check_same_name(self, node, counter):
        name = node.GetName()
        if name in counter:
            counter[name] += 1
            new_name = name + f"_{counter[name]}"
            node.SetName(new_name)
            print(f"Node name changed: {name} -> {new_name}")
        else:
            counter[name] = 0
        
        for i in range(node.GetChildCount()):
            self.check_same_name(node.GetChild(i), counter)
    
    def get_scene_keyframes(self, scale):
        keyframes = []
        criteria = FbxCriteria.ObjectType(FbxAnimStack.ClassId)
        for i in range(self.scene.GetSrcObjectCount(criteria)):
            anim_stack = self.scene.GetSrcObject(criteria, i)
            
            scene_kf = fbx_animation.get_scene_animation(anim_stack, self.scene.GetRootNode(), scale)
            keyframes.append(scene_kf)
        
        return keyframes
    
    def get_scene_fps(self):
        return fbx.FbxTime.GetFrameRate(self.scene.GetGlobalSettings().GetTimeMode())

class SkinningData:
    def __init__(self):
        self.name_to_idx    = {}
        self.joint_order    = []
        self.offset_xform   = []

        self.joint_indices1 = []
        self.joint_weights1 = []
        self.joint_indices2 = []
        self.joint_weights2 = []

class MeshData:
    def __init__(self):
        self.control_point_idx_to_vertex_idx = {}
        self.indices                         = []
        self.positions                       = []
        self.normals                         = []
        self.uvs                             = []
        self.tangents                        = []
        self.bitangents                      = []

        self.is_skinned: bool                = False
        self.skinning_data: SkinningData     = SkinningData()

        self.materials: list[MaterialInfo]   = []
        self.textures: list[TextureInfo]     = []

        self.polygon_material_connection     = []

class JointData:
    def __init__(self):
        self.name       = ""
        self.parent_idx = -1

        self.local_T    = glm.vec3(0.0)
        self.local_R    = glm.quat(1.0, 0.0, 0.0, 0.0)
        self.pre_R      = glm.quat(1.0, 0.0, 0.0, 0.0)
        self.local_S    = glm.vec3(1.0)

class CharacterData:
    def __init__(self):
        self.name       = ""
        self.joint_data: list[JointData] = []

class TextureInfo:
    def __init__(self):
        """
        alpha_src:
            "None", "RGB Intensity", "Black"
        mapping_type:
            "Null", "Planar", "Spherical", "Cylindrical", "Box", "Face", "UV", "Environment"
        blend_mode:
            "Translucent", "Additive", "Modulate", "Modulate2", "Over", "Normal", "Dissolve", "Darken", "ColorBurn", "LinearBurn",
            "DarkerColor", "Lighten", "Screen", "ColorDodge", "LinearDodge", "LighterColor", "SoftLight", "HardLight", "VividLight",
            "LinearLight", "PinLight", "HardMix", "Difference", "Exclusion", "Substract", "Divide", "Hue", "Saturation", "Color",
            "Luminosity", "Overlay"
        material_use:
            "Model Material", "Default Material"
        texture_use:
            "Standard", "Shadow Map", "Light Map", "Spherical Reflexion Map", "Sphere Reflexion Map", "Bump Normal Map"
        """
        self.name                  = ""
        self.filename              = ""
        self.scale_u               = 1.0
        self.scale_v               = 1.0
        self.translation_u         = 0.0
        self.translation_v         = 0.0
        self.swap_uv               = False
        self.rotation_u            = 0.0
        self.rotation_v            = 0.0
        self.rotation_w            = 0.0
        self.alpha_source          = "None"
        self.crop_left             = 0.0
        self.crop_top              = 0.0
        self.crop_right            = 0.0
        self.crop_bottom           = 0.0
        self.mapping_type          = ""
        self.planar_mapping_normal = ""
        self.blend_mode            = ""
        self.alpha                 = 0.0
        self.material_use          = ""
        self.texture_use           = ""

        """
        property
            "DiffuseColor", "EmissiveColor", "SpecularColor", "ShininessExponent", "NormalMap"
        """
        self.property               = ""
        self.connected_material     = -1
        self.is_used                = False

class MaterialInfo:
    def __init__(self):
        self.name         = ""
        self.ambient      = glm.vec3(0.0)
        self.diffuse      = glm.vec3(0.0)
        self.specular     = glm.vec3(0.0)
        self.emissive     = glm.vec3(0.0)
        self.opacity      = 0.0
        self.shininess    = 0.0
        self.reflectivity = 0.0
        self.material_id  = -1
        self.type         = ""
        self.texture_ids  = []