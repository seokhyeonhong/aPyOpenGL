from enum import Enum
from OpenGL.GL import *
import glfw
import glm

from pymovis.vis.primitives import *
from pymovis.vis.material import Material
from pymovis.vis.shader import Shader
from pymovis.vis.primitives import Cube
from pymovis.vis.renderoption import RenderOptions
from pymovis.vis import glconst

class RenderMode(Enum):
    PHONG = 0
    SHADOW = 1

class RenderInfo:
    sky_color         = glm.vec4(0.5, 0.5, 0.5, 1.0)
    cam_position      = glm.vec3(0)
    cam_projection    = glm.mat4(1)
    cam_view          = glm.mat4(1)
    light_vector      = glm.vec4(0)
    light_color       = glm.vec3(1)
    light_attenuation = glm.vec3(0)
    light_matrix      = glm.mat4(1)

class RenderOptions:
    def __init__(
        self,
        mesh,
        shader,
        draw_func
    ):
        self._mesh        = mesh
        self._shader      = shader
        self._position    = glm.vec3(0)
        self._orientation = glm.mat3(1)
        self._scale       = glm.vec3(1)
        self._material    = Material()
        self._uv_repeat   = glm.vec2(1)
        self._draw_func   = draw_func
    
    def get_vao(self):
        return self._mesh.vao

    def get_vao_id(self):
        return self._mesh.vao.id
    
    def draw(self):
        self._draw_func(self, self._shader)

    def get_position(self):
        return self._position

    def get_orientation(self):
        return self._orientation

    def get_scale(self):
        return self._scale
    
    def get_texture_id(self):
        return self._material.get_albedo_map().get_texture_id()
    
    def get_uv_repeat(self):
        return self._uv_repeat

    def set_position(self, x, y=None, z=None):
        if y == None and z == None:
            self._position = glm.vec3(x)
        elif y != None and z != None:
            self._position = glm.vec3(x, y, z)
        return self

    def set_orientation(self, orientation):
        self._orientation = glm.mat3(orientation)
        return self
    
    def set_scale(self, x, y=None, z=None):
        if y == None and z == None:
            self._scale = glm.vec3(x)
        elif y != None and z != None:
            self._scale = glm.vec3(x, y, z)
        return self

    def set_material(self, albedo=None, diffuse=None, specular=None):
        if albedo != None:
            self._material.set_albedo(albedo)
        if diffuse != None:
            self._material.set_diffuse(diffuse)
        if specular != None:
            self._material.set_specular(specular)
        return self

    def set_texture(self, filename):
        self._material.set_texture(filename)
        return self
    
    def set_uv_repeat(self, u, v=None):
        if v == None:
            self._uv_repeat = glm.vec2(u)
        else:
            self._uv_repeat = glm.vec2(u, v)
        return self

class Render:
    render_mode = RenderMode.PHONG
    render_info = RenderInfo()

    @staticmethod
    def initialize_shaders():
        Render.primitive_shader = Shader("basic.vs", "basic.fs")
        Render.shadow_shader    = Shader("shadow.vs", "shadow.fs")
        Render._generate_shadow_buffer()
    
    @staticmethod
    def _generate_shadow_buffer():
        # create depth texture
        depth_map_fbo = glGenFramebuffers(1)
        depth_map = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, depth_map)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, glconst.SHADOW_MAP_SIZE, glconst.SHADOW_MAP_SIZE, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
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

        Render.depth_map = depth_map
        Render.depth_map_fbo = depth_map_fbo

    @staticmethod
    def sky_color():
        return Render.render_info.sky_color
    
    @staticmethod
    def set_render_mode(mode, width, height):
        if mode == RenderMode.SHADOW:
            glBindFramebuffer(GL_FRAMEBUFFER, Render.depth_map_fbo)
        else:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
        Render.render_mode = mode

    @staticmethod
    def render_options(p):
        if Render.render_mode == RenderMode.SHADOW:
            return RenderOptions(p, Render.shadow_shader, Render.draw_shadow)
        else:
            return RenderOptions(p, Render.primitive_shader, Render.draw_phong)

    # TODO: implement this
    # @staticmethod
    # def render_options_vec(vec):
    #     if Render.render_mode == RenderMode.SHADOW:
    #         return RenderOptions(vec, Render.shadow_shader, Render.draw_shadow)
    #     else:
    #         return RenderOptions(vec, Render.primitive_shader, Render.draw_phong)

    @staticmethod
    def cube():
        if not hasattr(Render, "_cube"):
            Render._cube = Cube()
        return Render.render_options(Render._cube)

    @staticmethod
    def sphere(radius=0.5, stacks=32, sectors=32):
        if not hasattr(Render, "_sphere"):
            Render._sphere = Sphere(radius, stacks, sectors)
        return Render.render_options(Render._sphere)
    
    @staticmethod
    def plane():
        if not hasattr(Render, "_plane"):
            Render._plane = Plane()
        return Render.render_options(Render._plane)

    @staticmethod
    def draw_phong(option: RenderOptions, shader: Shader):
        if option == None or shader == None:
            return

        shader.use()
        
        # update view
        shader.set_mat4("P", Render.render_info.cam_projection)
        shader.set_mat4("V", Render.render_info.cam_view)
        shader.set_vec3("viewPosition", Render.render_info.cam_position)
        shader.set_vec4("light.vector", Render.render_info.light_vector)
        shader.set_vec3("light.color", Render.render_info.light_color)
        shader.set_vec3("light.attenuation", Render.render_info.light_attenuation)
        shader.set_mat4("lightSpaceMatrix", Render.render_info.light_matrix)

        # update model
        T = glm.translate(glm.mat4(1.0), option.get_position())
        R = glm.mat4(option.get_orientation())
        S = glm.scale(glm.mat4(1.0), option.get_scale())
        transform = T * R * S
        shader.set_mat4("M", transform)

        # set textures
        shader.set_int("material.albedoMap", 0)
        if option.get_texture_id() != None:
            shader.set_int("material.id", 0)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, option.get_texture_id())
        else:
            shader.set_int("material.id", -1)
        
        shader.set_int("shadowMap", 1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, Render.depth_map)
            
        # set material
        shader.set_vec3("material.diffuse", option._material.get_diffuse())
        shader.set_vec3("material.specular", option._material.get_specular())
        shader.set_float("material.shininess", option._material.get_shininess())
        shader.set_vec3("material.albedo", option._material.get_albedo())

        shader.set_vec2("uvScale", option.get_uv_repeat())

        # final rendering
        glBindVertexArray(option.get_vao_id())
        glDrawElements(GL_TRIANGLES, option.get_vao().indices_count, GL_UNSIGNED_INT, None)

        # unbind vao
        glBindVertexArray(0)
    
    @staticmethod
    def draw_shadow(option: RenderOptions, shader: Shader):
        if option == None or shader == None:
            return
        
        shader.use()

        shader.set_mat4("lightSpaceMatrix", Render.render_info.light_matrix)
        T = glm.translate(glm.mat4(1.0), option.get_position())
        R = glm.mat4(option.get_orientation())
        S = glm.scale(glm.mat4(1.0), option.get_scale())
        transform = T * R * S
        shader.set_mat4("M", transform)

        # final rendering
        glCullFace(GL_FRONT)
        glBindVertexArray(option.get_vao_id())
        glDrawElements(GL_TRIANGLES, option.get_vao().indices_count, GL_UNSIGNED_INT, None)
        glCullFace(GL_BACK)

        # unbind vao
        glBindVertexArray(0)

    @staticmethod
    def update_render_view(app, width, height):
        cam = app.camera
        light = app.light

        Render.render_info.cam_position      = cam.position
        Render.render_info.cam_projection    = cam.get_projection_matrix(width, height)
        Render.render_info.cam_view          = cam.get_view_matrix()
        Render.render_info.light_vector      = light.vector
        Render.render_info.light_color       = light.color * light.intensity
        Render.render_info.light_attenuation = light.attenuation
        Render.render_info.light_matrix      = light.get_view_projection_matrix()