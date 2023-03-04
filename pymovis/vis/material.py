import glm

from pymovis.vis.texture import Texture, TextureType

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
    
    def set_texture(self, texture, texture_type):
        texture_type = self.type_dict.get(texture_type, TextureType.eUNKNOWN)
        if texture_type == TextureType.eUNKNOWN:
            raise Exception("Texture type not supported")

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