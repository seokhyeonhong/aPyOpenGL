import os
import numpy as np
import copy
from enum import Enum

from OpenGL.GL import *
import glm

from PIL import Image
import imageio

from pymovis.vis.primitives import Cube
from pymovis.vis.const import SHADOW_MAP_SIZE, BACKGROUND_MAP_SIZE

def get_texture_data(filename, channels="RGBA", hdr=False):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    texture_path = os.path.join(curr_path, "texture", filename)

    texture_image = imageio.imread(texture_path, format="HDR-FI" if hdr else None)
    texture_image = np.array(texture_image)
    texture_image = np.flipud(texture_image)
    
    if hdr:
        texture_data = texture_image.tobytes()
    else:
        texture_data = Image.fromarray(texture_image)
        texture_data = texture_data.convert(channels).tobytes()

    height, width = texture_image.shape[:2]
    return texture_data, height, width

def render_cube():
    vao = Cube()
    glBindVertexArray(vao.id)
    glDrawElements(GL_TRIANGLES, len(vao.indices), GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

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
    
    def __deepcopy__(self, memo):
        res = Texture()
        res.path = self.path
        res.texture_id = self.texture_id
        memo[id(self)] = res
        return res

""" All textures are loaded in this class """
class TextureLoader:
    # Singleton
    __instance = None
    __texture_map = {}
    __cubemap_texture_map = {}
    __hdr_texture_map = {}
    __irradiance_map = {}

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
    def create_hdr(path):
        if path in TextureLoader.__hdr_texture_map:
            glDeleteTextures(1, TextureLoader.__hdr_texture_map[path].texture_id)
            del TextureLoader.__hdr_texture_map[path]
        
        texture_id = TextureLoader.generate_hdr_texture(path)
        texture = Texture(path, texture_id)
        TextureLoader.__hdr_texture_map[path] = texture

        return texture
    
    @staticmethod
    def create_irradiance_map(path):
        if path in TextureLoader.__irradiance_map:
            glDeleteTextures(1, TextureLoader.__irradiance_map[path].texture_id)
            del TextureLoader.__irradiance_map[path]
        
        hdr_texture = TextureLoader.load_hdr(path)
        texture_id = TextureLoader.generate_irradiance_map(hdr_texture)
        texture = Texture(path, texture_id)
        TextureLoader.__irradiance_map[path] = texture

        return texture
        
    @staticmethod
    def load(path) -> Texture:
        if path not in TextureLoader.__texture_map:
            texture_id = TextureLoader.generate_texture(path)
            texture = Texture(path, texture_id)
            TextureLoader.__texture_map[path] = texture
            
        return TextureLoader.__texture_map[path]
    
    @staticmethod
    def load_cubemap(dirname) -> Texture:
        if dirname not in TextureLoader.__cubemap_texture_map:
            texture_id = TextureLoader.generate_cubemap_texture(dirname)
            texture = Texture(dirname, texture_id)
            TextureLoader.__cubemap_texture_map[dirname] = texture

        return TextureLoader.__cubemap_texture_map[dirname]
    
    @staticmethod
    def load_hdr(path) -> Texture:
        if path not in TextureLoader.__hdr_texture_map:
            texture_id = TextureLoader.generate_hdr_texture(path)
            texture = Texture(path, texture_id)
            TextureLoader.__hdr_texture_map[path] = texture

        return TextureLoader.__hdr_texture_map[path]
    
    @staticmethod
    def load_irradiance_map(path, equirect_to_cubemap_shader) -> Texture:
        if path not in TextureLoader.__irradiance_map:
            hdr_texture = TextureLoader.load_hdr(path)
            texture_id = TextureLoader.generate_irradiance_map(hdr_texture, equirect_to_cubemap_shader)
            texture = Texture(path, texture_id)
            TextureLoader.__irradiance_map[path] = texture

        return TextureLoader.__irradiance_map[path]

    @staticmethod
    def clear():
        for texture in TextureLoader.__texture_map.values():
            glDeleteTextures(1, np.array(texture.texture_id, dtype=np.uint32))
        TextureLoader.__texture_map.clear()

        for texture in TextureLoader.__cubemap_texture_map.values():
            glDeleteTextures(1, np.array(texture.texture_id, dtype=np.uint32))
        TextureLoader.__cubemap_texture_map.clear()

        for texture in TextureLoader.__hdr_texture_map.values():
            glDeleteTextures(1, np.array(texture.texture_id, dtype=np.uint32))
        TextureLoader.__hdr_texture_map.clear()

        for texture in TextureLoader.__irradiance_map.values():
            glDeleteTextures(1, np.array(texture.texture_id, dtype=np.uint32))
        TextureLoader.__irradiance_map.clear()
    
    @staticmethod
    def generate_texture(filename, nearest=False):
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        texture_data, height, width = get_texture_data(filename)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
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
        # HDR texture
        hdr_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, hdr_texture)
        
        texture_data, height, width = get_texture_data(filename, channels="RGB", hdr=True)

        # NOTE: GL_RGB16F instead of GL_RGBA for HDR textures
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, texture_data)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        return hdr_texture
    
    @staticmethod
    def generate_irradiance_map(hdr_texture, equirect_to_cubemap_shader):
        # configure capture framebuffer
        capture_fbo = glGenFramebuffers(1)
        capture_rbo = glGenRenderbuffers(1)

        glBindFramebuffer(GL_FRAMEBUFFER, capture_fbo)
        glBindRenderbuffer(GL_RENDERBUFFER, capture_rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, BACKGROUND_MAP_SIZE, BACKGROUND_MAP_SIZE)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, capture_rbo)

        # create cubemap to render to and attach to framebuffer
        irradiance_map = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, irradiance_map)
        for i in range(6):
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, BACKGROUND_MAP_SIZE, BACKGROUND_MAP_SIZE, 0, GL_RGB, GL_FLOAT, None)

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # projection and view matrices
        capture_projection = glm.perspective(glm.radians(90.0), 1.0, 0.1, 10.0)
        capture_views = [
            glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(1.0, 0.0, 0.0), glm.vec3(0.0, -1.0, 0.0)),
            glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(-1.0, 0.0, 0.0), glm.vec3(0.0, -1.0, 0.0)),
            glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0), glm.vec3(0.0, 0.0, 1.0)),
            glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, -1.0, 0.0), glm.vec3(0.0, 0.0, -1.0)),
            glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 0.0, 1.0), glm.vec3(0.0, -1.0, 0.0)),
            glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 0.0, -1.0), glm.vec3(0.0, -1.0, 0.0))
        ]

        # convert HDR equirectangular environment map to cubemap equivalent
        equirect_to_cubemap_shader.use()
        equirect_to_cubemap_shader.set_int("uEquirectangularMap", 0)
        equirect_to_cubemap_shader.set_mat4("uProjection", capture_projection)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, hdr_texture.texture_id)

        glViewport(0, 0, BACKGROUND_MAP_SIZE, BACKGROUND_MAP_SIZE) # don't forget to configure the viewport to the capture dimensions.
        glBindFramebuffer(GL_FRAMEBUFFER, capture_fbo)
        for i in range(6):
            equirect_to_cubemap_shader.set_mat4("uView", capture_views[i])
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, irradiance_map, 0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            render_cube()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        return irradiance_map

    @staticmethod
    def generate_shadow_buffer():
        # create depth texture
        shadowmap_fbo = glGenFramebuffers(1)
        shadowmap = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, shadowmap)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        border_color = [1.0, 1.0, 1.0, 1.0]
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color)

        # create frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, shadowmap_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowmap, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)

        # reset the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        return shadowmap_fbo, shadowmap