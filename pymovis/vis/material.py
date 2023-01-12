import glm

from pymovis.vis.texture import Texture, TextureType, CubemapTexture

class Material:
    def __init__(self,
        albedo=glm.vec3(0.5),
        diffuse=glm.vec3(1.0),
        specular=glm.vec3(0.1),
        shininess=10.0
    ):
        self.__albedo     = glm.vec3(albedo)
        self.__diffuse    = glm.vec3(diffuse)
        self.__specular   = glm.vec3(specular)
        self.__shininess  = shininess
        self.__alpha      = 1.0
        self.__albedo_map = Texture()
        self.__cubemap    = CubemapTexture()
    
    @property
    def albedo(self):
        return self.__albedo
    
    @property
    def diffuse(self):
        return self.__diffuse
    
    @property
    def specular(self):
        return self.__specular

    @property
    def albedo_map(self):
        return self.__albedo_map
    
    @property
    def cubemap(self):
        return self.__cubemap

    @property
    def shininess(self):
        return self.__shininess
    
    @property
    def alpha(self):
        return self.__alpha
    
    def set_texture(self, filename):
        # TODO: Add other types of maps (e.g. specular map, normal map, ...)
        self.__albedo_map.set_texture(filename)
    
    def set_cubemap(self, dirname):
        self.__cubemap.set_texture(dirname)

    def set_albedo(self, albedo):
        self.__albedo = glm.vec3(albedo)
    
    def set_diffuse(self, diffuse):
        self.__diffuse = glm.vec3(diffuse)
    
    def set_specular(self, specular):
        self.__specular = glm.vec3(specular)
    
    def set_shininess(self, shininess):
        self.__shininess = shininess
    
    def set_alpha(self, alpha):
        self.__alpha = alpha