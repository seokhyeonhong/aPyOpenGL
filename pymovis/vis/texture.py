import os
from OpenGL.GL import *
import numpy as np
from PIL import Image
from enum import Enum

from pymovis.vis.const import SHADOW_MAP_SIZE

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

""" Texture info """
class Texture:
    def __init__(self, path=None, texture_id=0):
        self.texture_loader = TextureLoader()

        self.path = path
        self.texture_id = texture_id

""" All textures are loaded in this class """
class TextureLoader:
    # Singleton
    __instance = None
    __texture_map = {}
    __cubemap_texture_map = {}
    __hdr_texture_map = {}

    def __new__(cls):
        if TextureLoader.__instance is None:
            TextureLoader.__instance = object.__new__(cls)
        return TextureLoader.__instance
    
    def __init__(self):
        pass

    @staticmethod
    def create(path, nearest=False):
        if path in TextureLoader.__texture_map:
            glDeleteTextures(1, TextureLoader.__texture_map[path].texture_id)
            del TextureLoader.__texture_map[path]
        
        texture_id = TextureLoader.generate_texture(path, nearest)
        texture = Texture(path, texture_id)
        TextureLoader.__texture_map[path] = texture

        return texture

    @staticmethod
    def create_cubemap(dirname):
        if dirname in TextureLoader.__cubemap_texture_map:
            glDeleteTextures(1, TextureLoader.__cubemap_texture_map[dirname].texture_id)
            del TextureLoader.__cubemap_texture_map[dirname]
        
        texture_id = TextureLoader.generate_cubemap_texture(dirname)
        texture = Texture(dirname, texture_id)
        TextureLoader.__cubemap_texture_map[dirname] = texture

        return texture
    
    @staticmethod
    def load(path):
        if path not in TextureLoader.__texture_map:
            texture_id = TextureLoader.generate_texture(path)
            texture = Texture(path, texture_id)
            TextureLoader.__texture_map[path] = texture
            
        return TextureLoader.__texture_map[path]
    
    @staticmethod
    def load_cubemap(dirname):
        if dirname not in TextureLoader.__cubemap_texture_map:
            texture_id = TextureLoader.generate_cubemap_texture(dirname)
            texture = Texture(dirname, texture_id)
            TextureLoader.__cubemap_texture_map[dirname] = texture

        return TextureLoader.__cubemap_texture_map[dirname]

    @staticmethod
    def clear():
        for texture in TextureLoader.__texture_map.values():
            glDeleteTextures(1, np.array(texture.texture_id, dtype=np.uint32))
        TextureLoader.__texture_map.clear()

        for texture in TextureLoader.__cubemap_texture_map.values():
            glDeleteTextures(1, np.array(texture.texture_id, dtype=np.uint32))
        TextureLoader.__cubemap_texture_map.clear()
    
    @staticmethod
    def generate_texture(filename, nearest=False):
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
        
        return texture_id

    @staticmethod
    def generate_cubemap_texture(dirname):
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
    
    @staticmethod
    def generate_hdr_texture(filename):
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        curr_path = os.path.dirname(os.path.abspath(__file__))
        texture_path = os.path.join(curr_path, "texture", filename)
        texture_image = Image.open(texture_path)
        texture_image = texture_image.transpose(Image.FLIP_TOP_BOTTOM)
        texture_data = texture_image.convert("RGBA").tobytes()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 512, 512, 0, GL_RGB, GL_FLOAT, None)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        return texture_id
    
    @staticmethod
    def generate_shadow_buffer():
        # create depth texture
        depth_map_fbo = glGenFramebuffers(1)
        depth_map = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, depth_map)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        border_color = [1.0, 1.0, 1.0, 1.0]
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color)

        # create frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, depth_map_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_map, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)

        # reset the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        return depth_map_fbo, depth_map