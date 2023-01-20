import glm

from pymovis.vis.texture import Texture, TextureType

class Material:
    def __init__(self,
        albedo:    glm.vec3 = glm.vec3(0.5),
        diffuse:   glm.vec3 = glm.vec3(1.0),
        specular:  glm.vec3 = glm.vec3(0.1),
        shininess: float = 10.0
    ):
        self.albedo     = glm.vec3(albedo)
        self.diffuse    = glm.vec3(diffuse)
        self.specular   = glm.vec3(specular)
        self.shininess  = shininess
        self.alpha      = 1.0
        self.albedo_map = Texture()
        self.cubemap    = Texture()
    
    def set_texture(self, texture, texture_type):
        if texture_type == TextureType.eALBEDO or texture_type == TextureType.eDIFFUSE:
            self.albedo_map = texture
        elif texture_type == TextureType.eCUBE_MAP:
            self.cubemap = texture
        else:
            raise Exception("Texture type not supported")

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