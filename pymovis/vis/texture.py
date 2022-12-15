import os
from OpenGL.GL import *
from PIL import Image
from enum import Enum

class TextureType(Enum):
    ALBEDO = 0

class Texture:
    texture_id_map = {} # texture map to avoid loading the same texture twice
    
    def __init__(
        self,
        filename=None
    ):
        self.__texture_id = Texture.texture_id_map[filename] if filename in Texture.texture_id_map else None

    def set_texture(self, filename, nearest=False):
        if filename in Texture.texture_id_map:
            self.__texture_id = Texture.texture_id_map[filename]
        else:
            self.__texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.__texture_id)

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
            
            Texture.texture_id_map[filename] = self.__texture_id

    @property
    def texture_id(self):
        return self.__texture_id
    
    @staticmethod
    def clear():
        Texture.texture_id_map.clear()

class CubemapTexture:
    """
    Texture for cubemap
    All the texture faces should be in the same directory and named as "right", "left", "top", "bottom", "front", "back"
    Also, the texture faces should be in jpg format
    """
    texture_id_map = {} # texture map to avoid loading the same texture twice
    def __init__(
        self,
        dirname=None
    ):
        self.__texture_id = None
    
    def set_texture(self, dirname):
        if dirname in CubemapTexture.texture_id_map:
            self.__texture_id = CubemapTexture.texture_id_map[dirname]
        else:
            self.__texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_CUBE_MAP, self.__texture_id)

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

            CubemapTexture.texture_id_map[dirname] = self.__texture_id

    @property
    def texture_id(self):
        return self.__texture_id
    
    @staticmethod
    def clear():
        CubemapTexture.texture_id_map.clear()