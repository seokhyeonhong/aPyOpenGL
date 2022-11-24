import os
from OpenGL.GL import *
from PIL import Image
from enum import Enum

class TextureType(Enum):
    ALBEDO = 0

class Texture:
    texture_id_map = {}
    def __init__(
        self,
        filename=None
    ):
        self._filename = filename
        self._texture_id = self.set_texture(filename, False) if filename != None else None

    def set_texture(self, filename, nearest=False):
        if filename in Texture.texture_id_map:
            self._texture_id = Texture.texture_id_map[filename]
        else:
            self._texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self._texture_id)

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
            
            Texture.texture_id_map[filename] = self._texture_id

    def get_texture_id(self):
        return self._texture_id
    
    @staticmethod
    def clear():
        Texture.texture_id_map.clear()