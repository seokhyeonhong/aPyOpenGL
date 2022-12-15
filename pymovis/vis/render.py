from __future__ import annotations
from enum import Enum

from OpenGL.GL import *
import glfw
import glm

from pymovis.vis.primitives import *
from pymovis.vis.material import Material
from pymovis.vis.shader import Shader
from pymovis.vis.primitives import Cube
# from pymovis.vis.renderoption import RenderOptions, RenderOptionsVec
from pymovis.vis.texture import Texture
from pymovis.vis.text import FontTexture
from pymovis.vis import glconst

class RenderMode(Enum):
    PHONG  = 0
    SHADOW = 1

class RenderInfo:
    sky_color         = glm.vec4(1.0)
    cam_position      = glm.vec3(0)
    cam_projection    = glm.mat4(1)
    cam_view          = glm.mat4(1)
    light_vector      = glm.vec4(0)
    light_color       = glm.vec3(1)
    light_attenuation = glm.vec3(0)
    light_matrix      = glm.mat4(1)

class Render:
    """
    Global rendering state and functions
    """
    render_mode = RenderMode.PHONG
    render_info = RenderInfo()
    primitive_meshes = {}
    font_texture = None

    @staticmethod
    def initialize_shaders():
        Render.primitive_shader = Shader("phong.vs", "phong.fs")
        Render.shadow_shader    = Shader("shadow.vs", "shadow.fs")
        Render.text_shader      = Shader("text.vs", "text.fs")
        Render.cubemap_shader   = Shader("cubemap.vs", "cubemap.fs")
        Render.generate_shadow_buffer()
    
    @staticmethod
    def generate_shadow_buffer():
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

    @staticmethod
    def cube():
        return RenderOptions(Cube(), Render.primitive_shader, Render.draw_phong, Render.shadow_shader, Render.draw_shadow)

    @staticmethod
    def sphere(radius=0.5, stacks=32, sectors=32):
        return RenderOptions(Sphere(radius, stacks, sectors), Render.primitive_shader, Render.draw_phong, Render.shadow_shader, Render.draw_shadow)
    
    @staticmethod
    def cone(radius=0.5, height=1, sectors=16):
        return RenderOptions(Cone(radius, height, sectors), Render.primitive_shader, Render.draw_phong, Render.shadow_shader, Render.draw_shadow)

    @staticmethod
    def plane():
        return RenderOptions(Plane(), Render.primitive_shader, Render.draw_phong, Render.shadow_shader, Render.draw_shadow)
    
    @staticmethod
    def cylinder(radius=0.5, height=1, sectors=16):
        return RenderOptions(Cylinder(radius, height, sectors), Render.primitive_shader, Render.draw_phong, Render.shadow_shader, Render.draw_shadow)

    @staticmethod
    def arrow():
        cone_radius = 0.07
        cone_height = 0.2
        cylinder_radius = 0.03
        cylinder_height = 0.8

        R_x = glm.rotate(glm.mat4(1.0), glm.radians(-90), glm.vec3(0, 0, 1))
        x_head = RenderOptions(Cone(cone_radius, cone_height, 16), Render.primitive_shader, Render.draw_phong).set_position(0.9, 0, 0).set_orientation(R_x).set_material(albedo=glm.vec3(1, 0, 0)).set_color_mode(True)
        x_body = RenderOptions(Cylinder(cylinder_radius, cylinder_height, 16), Render.primitive_shader, Render.draw_phong).set_position(0.4, 0, 0).set_orientation(R_x).set_material(albedo=glm.vec3(1, 0, 0)).set_color_mode(True)

        y_head = RenderOptions(Cone(cone_radius, cone_height, 16), Render.primitive_shader, Render.draw_phong).set_position(0, 0.9, 0).set_material(albedo=glm.vec3(0, 1, 0)).set_color_mode(True)
        y_body = RenderOptions(Cylinder(cylinder_radius, cylinder_height, 16), Render.primitive_shader, Render.draw_phong).set_position(0, 0.4, 0).set_material(albedo=glm.vec3(0, 1, 0)).set_color_mode(True)

        R_z = glm.rotate(glm.mat4(1.0), glm.radians(90), glm.vec3(1, 0, 0))
        z_head = RenderOptions(Cone(cone_radius, cone_height, 16), Render.primitive_shader, Render.draw_phong).set_position(0, 0, 0.9).set_orientation(R_z).set_material(albedo=glm.vec3(0, 0, 1)).set_color_mode(True)
        z_body = RenderOptions(Cylinder(cylinder_radius, cylinder_height, 16), Render.primitive_shader, Render.draw_phong).set_position(0, 0, 0.4).set_orientation(R_z).set_material(albedo=glm.vec3(0, 0, 1)).set_color_mode(True)
        return RenderOptionsVec([x_head, x_body, y_head, y_body, z_head, z_body])

    @staticmethod
    def text(t):
        if Render.font_texture is None:
            Render.font_texture = FontTexture()

        if Render.render_mode == RenderMode.SHADOW:
            return RenderOptions(VAO(), None, Render.draw_shadow)
        else:
            res = RenderOptions(VAO(), Render.text_shader, Render.draw_text)
            res.set_text(str(t))
            res.set_material(albedo=glm.vec3(0))
            return res

    @staticmethod
    def cubemap(dirname, scale=100):
        ro = RenderOptions(Cubemap(scale=scale), Render.cubemap_shader, Render.draw_cubemap)
        ro.set_cubemap(dirname)
        return ro

    @staticmethod
    def draw_phong(option: RenderOptions, shader: Shader):
        if option is None or shader is None:
            return
        if Render.render_mode == RenderMode.SHADOW:
            return

        shader.use()
        
        # update view
        shader.set_mat4("P",                  Render.render_info.cam_projection)
        shader.set_mat4("V",                  Render.render_info.cam_view)
        shader.set_vec3("viewPosition",       Render.render_info.cam_position)
        shader.set_vec4("uLight.vector",      Render.render_info.light_vector)
        shader.set_vec3("uLight.color",       Render.render_info.light_color)
        shader.set_vec3("uLight.attenuation", Render.render_info.light_attenuation)
        shader.set_mat4("lightSpaceMatrix",   Render.render_info.light_matrix)

        # update model
        T = glm.translate(glm.mat4(1.0), option.position)
        R = glm.mat4(option.orientation)
        S = glm.scale(glm.mat4(1.0), option.scale)
        transform = T * R * S
        shader.set_mat4("M", transform)

        # set textures
        shader.set_int("uMaterial.albedoMap", 0)
        if option.texture_id is not None:
            shader.set_int("uMaterial.id", 0)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, option.texture_id)
        else:
            shader.set_int("uMaterial.id", -1)
        
        shader.set_int("uShadowMap", 1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, Render.depth_map)
            
        # set material
        shader.set_bool("uColorMode",           option.color_mode)
        shader.set_vec3("uMaterial.diffuse",    option.material.diffuse)
        shader.set_vec3("uMaterial.specular",   option.material.specular)
        shader.set_float("uMaterial.shininess", option.material.shininess)
        shader.set_vec3("uMaterial.albedo",     option.material.albedo)
        shader.set_float("uMaterial.alpha",     option.material.alpha)
        shader.set_vec2("uvScale",              option.uv_repeat)

        # final rendering
        glBindVertexArray(option.vao.id)
        glDrawElements(GL_TRIANGLES, option.vao.indices_count, GL_UNSIGNED_INT, None)

        # unbind vao
        glBindVertexArray(0)
    
    @staticmethod
    def draw_shadow(option: RenderOptions, shader: Shader):
        if option is None or shader is None:
            return
        if Render.render_mode != RenderMode.SHADOW:
            return
        
        shader.use()

        shader.set_mat4("lightSpaceMatrix", Render.render_info.light_matrix)
        T = glm.translate(glm.mat4(1.0), option.position)
        R = glm.mat4(option.orientation)
        S = glm.scale(glm.mat4(1.0), option.scale)
        transform = T * R * S
        shader.set_mat4("M", transform)

        # final rendering
        glCullFace(GL_FRONT)
        glBindVertexArray(option.vao.id)
        glDrawElements(GL_TRIANGLES, option.vao.indices_count, GL_UNSIGNED_INT, None)
        glCullFace(GL_BACK)

        # unbind vao
        glBindVertexArray(0)

    @staticmethod
    def draw_text(option: RenderOptions, shader: Shader):
        if option is None or shader is None:
            return
            
        x = 0
        y = 0
        scale = option.scale.x / glconst.TEXT_RESOLUTION

        # shader settings
        shader.use()
        
        if option.text_fixed:
            # TODO: implement here
            raise NotImplementedError()
        else:
            shader.set_mat4("P", Render.render_info.cam_projection)
            shader.set_mat4("V", Render.render_info.cam_view)

            T = glm.translate(glm.mat4(1.0), option.position)
            R = glm.mat4(option.orientation)
            S = glm.scale(glm.mat4(1.0), option.scale)
            transform = T * R * S
            shader.set_mat4("M", transform)

        shader.set_int("uText", 0)
        shader.set_vec3("uTextColor", option.material.albedo)

        glActiveTexture(GL_TEXTURE0)
        glBindVertexArray(Render.font_texture.vao)

        for c in option.text:
            ch = Render.font_texture.character(c)

            xpos = x + ch.bearing.x * scale
            ypos = y - (ch.size.y - ch.bearing.y) * scale

            w = ch.size.x * scale
            h = ch.size.y * scale

            vertices = np.array([
                xpos,     ypos + h,   0.0, 0.0,
                xpos,     ypos,       0.0, 1.0,
                xpos + w, ypos,       1.0, 1.0,

                xpos,     ypos + h,   0.0, 0.0,
                xpos + w, ypos,       1.0, 1.0,
                xpos + w, ypos + h,   1.0, 0.0
            ], dtype=np.float32)

            # render glyph texture
            glBindTexture(GL_TEXTURE_2D, ch.texture_id)

            # update content of VBO memory
            glBindBuffer(GL_ARRAY_BUFFER, Render.font_texture.vbo)
            glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

            # render quad
            glDrawArrays(GL_TRIANGLES, 0, 6)

            x += (ch.advance >> 6) * scale
        
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)

    @staticmethod
    def draw_cubemap(option: RenderOptions, shader: Shader):
        if option is None or shader is None:
            return
        if Render.render_mode == RenderMode.SHADOW:
            return
        
        # adjust depth settings for optimized rendering
        glDepthFunc(GL_LEQUAL)

        shader.use()

        # update view
        shader.set_mat4("P", Render.render_info.cam_projection)
        shader.set_mat4("V", glm.mat4(glm.mat3(Render.render_info.cam_view)))

        # set textures
        shader.set_int("uSkybox", 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, option.cubemap_id)

        # final rendering
        glBindVertexArray(option.vao.id)
        glDrawArrays(GL_TRIANGLES, 0, 36)

        # restore depth settings
        glDepthFunc(GL_LESS)

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
    
    @staticmethod
    def clear():
        Render.primitive_meshes.clear()
        Texture.clear()
        Render.font_texture = None

class RenderOptions:
    """
    Rendering options for a primitive (e.g. position, orientation, material, etc.)
    """
    def __init__(
        self,
        mesh,
        shader,
        draw_func,
        shadow_shader=None,
        shadow_func=None,
    ):
        self.mesh          = mesh
        self.shader        = shader
        self.shadow_shader = shadow_shader

        # transformation
        self.position      = glm.vec3(0)
        self.orientation   = glm.mat3(1)
        self.scale         = glm.vec3(1)

        # material  
        self.material      = Material()
        self.uv_repeat     = glm.vec2(1)
        self.text          = ""
        self.text_fixed    = False
        self.color_mode    = False

        self.draw_func     = draw_func
        self.shadow_func   = shadow_func

    @property
    def vao(self):
        return self.mesh.vao

    @property
    def texture_id(self):
        return self.material.albedo_map.texture_id
    
    @property
    def cubemap_id(self):
        return self.material.cubemap.texture_id

    def draw(self):
        if Render.render_mode == RenderMode.SHADOW:
            if self.shadow_func is not None:
                self.shadow_func(self, self.shadow_shader)
        else:
            self.draw_func(self, self.shader)

    def set_position(self, x, y=None, z=None):
        if y is None and z is None:
            self.position = glm.vec3(x)
        elif y != None and z != None:
            self.position = glm.vec3(x, y, z)
        return self

    def set_orientation(self, orientation):
        self.orientation = glm.mat3(orientation)
        return self
    
    def set_scale(self, x, y=None, z=None):
        if y is None and z is None:
            self.scale = glm.vec3(x)
        elif y != None and z != None:
            self.scale = glm.vec3(x, y, z)
        return self

    def set_material(self, albedo=None, diffuse=None, specular=None):
        if albedo != None:
            self.material.set_albedo(albedo)
        if diffuse != None:
            self.material.set_diffuse(diffuse)
        if specular != None:
            self.material.set_specular(specular)
        return self

    def set_texture(self, filename):
        self.material.set_texture(filename)
        return self
    
    def set_cubemap(self, dirname):
        self.material.set_cubemap(dirname)
        return self
    
    def set_uv_repeat(self, u, v=None):
        if v is None:
            self.uv_repeat = glm.vec2(u)
        else:
            self.uv_repeat = glm.vec2(u, v)
        return self
    
    def set_text(self, text: str):
        self.text = text
        return self
    
    def set_alpha(self, alpha):
        self.material.set_alpha(alpha)
        return self
    
    def set_color_mode(self, color_mode):
        self.color_mode = color_mode
        return self

class RenderOptionsVec:
    """
    Multiple rendering options for a primitive
    """
    def __init__(self, options: list[RenderOptions]):
        self.options = options
    
    def draw(self):
        for option in self.options:
            option.draw()
