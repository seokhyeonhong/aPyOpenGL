import glm
import copy

from .texture import Texture, TextureType

class Material:
    def __init__(self,
        albedo    = glm.vec3(0.5),
        diffuse   = glm.vec3(1.0),
        specular  = glm.vec3(0.1),
        shininess = 10.0
    ):
        self.albedo     = glm.vec3(albedo)
        self.alpha      = 1.0

        # phong model
        self.diffuse    = glm.vec3(diffuse)
        self.specular   = glm.vec3(specular)
        self.shininess  = shininess

        # pbr model
        self.metallic   = 0.0
        self.roughness  = 1.0
        self.ao         = 1.0

        # textures
        self.albedo_map    = Texture()
        self.normal_map    = Texture()
        self.disp_map      = Texture()
        self.metallic_map  = Texture()
        self.roughness_map = Texture()
        self.ao_map        = Texture()

        self.type_dict  = {
            "albedo"    : TextureType.eALBEDO,
            "diffuse"   : TextureType.eDIFFUSE,
            "normal"    : TextureType.eNORMAL,
            "disp"      : TextureType.eDISPLACEMENT,
            "metallic"  : TextureType.eMETALIC,
            "roughness" : TextureType.eROUGHNESS,
            "ao"        : TextureType.eAO
        }
    
    @staticmethod
    def from_mtl_dict(d):
        res = Material()

        value = d.get("ambient", None)
        if value is not None:
            res.albedo = glm.vec3(value)
        
        value = d.get("diffuse", None)
        if value is not None:
            res.diffuse = glm.vec3(value)

        value = d.get("specular", None)
        if value is not None:
            res.specular = glm.vec3(value)

        value = d.get("shininess", None)
        if value is not None:
            res.shininess = float(value)

    def __deepcopy__(self, memo):
        res = Material()

        res.albedo        = copy.deepcopy(self.albedo)
        res.alpha         = self.alpha

        res.diffuse       = copy.deepcopy(self.diffuse)
        res.specular      = copy.deepcopy(self.specular)
        res.shininess     = self.shininess

        res.metallic      = self.metallic
        res.roughness     = self.roughness
        res.ao            = self.ao

        res.albedo_map    = copy.deepcopy(self.albedo_map)
        res.normal_map    = copy.deepcopy(self.normal_map)
        res.disp_map      = copy.deepcopy(self.disp_map)
        res.metallic_map  = copy.deepcopy(self.metallic_map)
        res.roughness_map = copy.deepcopy(self.roughness_map)
        res.ao_map        = copy.deepcopy(self.ao_map)

        memo[id(self)] = res
        return res
    
    def set_texture(self, texture, texture_type):
        if not isinstance(texture_type, TextureType):
            texture_type = self.type_dict.get(texture_type, TextureType.eDIFFUSE)

        # TextureType: Unknown
        if texture_type == TextureType.eUNKNOWN:
            raise Exception("Texture type not supported")

        # TextureType
        if texture_type == TextureType.eALBEDO or texture_type == TextureType.eDIFFUSE:
            self.albedo_map = texture
        elif texture_type == TextureType.eNORMAL:
            self.normal_map = texture
        elif texture_type == TextureType.eDISPLACEMENT:
            self.disp_map = texture
        elif texture_type == TextureType.eMETALIC:
            self.metallic_map = texture
        elif texture_type == TextureType.eROUGHNESS:
            self.roughness_map = texture
        elif texture_type == TextureType.eAO:
            self.ao_map = texture

    def set_albedo(self, albedo):
        self.albedo = glm.vec3(albedo)
    
    def set_diffuse(self, diffuse):
        self.diffuse = glm.vec3(diffuse)
    
    def set_specular(self, specular):
        self.specular = glm.vec3(specular)
    
    def set_shininess(self, shininess):
        self.shininess = shininess
    
    def set_alpha(self, alpha):
        self.alpha = alpha