import os
from OpenGL.GL import *
from PIL import Image
from enum import Enum

class TextureType(Enum):
    eUNKNOWN      = -1
    eALBEDO       = 0
    eNORMAL       = 1
    eDIFFUSE      = 2
    eSPECULAR     = 3
    eMETALIC      = 4
    eROUGHNESS    = 5
    eAO           = 6
    eDISPLACEMENT = 7
    eEMISSIVE     = 8
    eGLOSSINESS   = 9
    eCUBEMAP      = 10

""" Texture management class """
class Texture:
    def __init__(self, path=None, texture_id=None):
        self.texture_loader = TextureLoader()

        self.path = path
        self.texture_id = texture_id
    
    def create(self, path, nearest=False):
        return self.texture_loader.create(path, nearest)
    
    def create_cubemap(self, dirname):
        return self.texture_loader.create_cubemap(dirname)

    def load(self, path):
        return self.texture_loader.load(path)
    
    def load_cubemap(self, dirname):
        return self.texture_loader.load_cubemap(dirname)
    
    @staticmethod
    def clear():
        TextureLoader().clear()

""" All texture files are loaded in this singleton class """
class TextureLoader:
    __instance = None
    def __new__(cls):
        if TextureLoader.__instance is None:
            TextureLoader.__instance = object.__new__(cls)
        return TextureLoader.__instance
    
    def __init__(self):
        self.__texture_map = {}
        self.__cubemap_texture_map = {}
    
    def create(self, path, nearest=False):
        if path in self.__texture_map:
            glDeleteTextures(1, self.__texture_map[path].texture_id)
            del self.__texture_map[path]
        
        texture_id, width, height, channel = self.generate_texture(path, nearest)
        texture = Texture(path, texture_id)
        self.__texture_map[path] = texture

        return texture

    def create_cubemap(self, dirname):
        if dirname in self.__cubemap_texture_map:
            glDeleteTextures(1, self.__cubemap_texture_map[dirname].texture_id)
            del self.__cubemap_texture_map[dirname]
        
        texture_id = self.generate_cubemap_texture(dirname)
        texture = Texture(dirname, texture_id)
        self.__cubemap_texture_map[dirname] = texture

        return texture

    def load(self, path):
        if path in self.__texture_map:
            return self.__texture_map[path]
        else:
            return self.create(path)

    def load_cubemap(self, path):
        if path in self.__cubemap_texture_map:
            return self.__cubemap_texture_map[path]
        else:
            return self.create_cubemap(path)

    def generate_texture(self, filename, nearest=False):
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        curr_path = os.path.dirname(os.path.abspath(__file__))
        texture_path = os.path.join(curr_path, "texture", filename)
        texture_image = Image.open(texture_path)
        texture_image = texture_image.transpose(Image.FLIP_TOP_BOTTOM)
        texture_data = texture_image.convert("RGBA").tobytes()

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture_image.width, texture_image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        if nearest:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        return texture_id, texture_image.width, texture_image.height, 4

    def generate_cubemap_texture(self, dirname):
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id)

        texture_faces = ["right", "left", "top", "bottom", "front", "back"]
        curr_path = os.path.dirname(os.path.abspath(__file__))
        texture_path = os.path.join(curr_path, "texture", dirname)
        for i, face in enumerate(texture_faces):
            # TODO: support all file extensions, not just jpg
            texture_image = Image.open(os.path.join(texture_path, face + ".jpg"))
            # texture_image = texture_image.transpose(Image.FLIP_TOP_BOTTOM)
            texture_data = texture_image.convert("RGBA").tobytes()

            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA, texture_image.width, texture_image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        return texture_id
    
    def clear(self):
        for texture in self.__texture_map.values():
            glDeleteTextures(1, texture.texture_id)
        self.__texture_map.clear()

        for texture in self.__cubemap_texture_map.values():
            glDeleteTextures(1, texture.texture_id)
        self.__cubemap_texture_map.clear()