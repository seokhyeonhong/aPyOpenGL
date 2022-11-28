import glm

from pymovis.vis.texture import Texture, TextureType

class Material:
    def __init__(self,
        albedo=glm.vec3(0.7, 0.0, 0.0),
        diffuse=glm.vec3(1.0),
        specular=glm.vec3(0.1),
        shininess=10.0
    ):
        self._albedo     = glm.vec3(albedo)
        self._diffuse    = glm.vec3(diffuse)
        self._specular   = glm.vec3(specular)
        self._shininess  = shininess
        self._alpha      = 1.0
        self._albedo_map = Texture()
    
    def get_albedo(self):
        return self._albedo
    
    def get_diffuse(self):
        return self._diffuse
    
    def get_specular(self):
        return self._specular

    def get_albedo_map(self):
        return self._albedo_map
    
    def get_shininess(self):
        return self._shininess
    
    def get_alpha(self):
        return self._alpha
    
    def set_texture(self, filename):
        # TODO: Add other types of maps (e.g. specular map, normal map, ...)
        self._albedo_map.set_texture(filename)
    
    def set_albedo(self, albedo):
        self._albedo = glm.vec3(albedo)
    
    def set_diffuse(self, diffuse):
        self._diffuse = glm.vec3(diffuse)
    
    def set_specular(self, specular):
        self._specular = glm.vec3(specular)
    
    def set_shininess(self, shininess):
        self._shininess = shininess
    
    def set_alpha(self, alpha):
        self._alpha = alpha