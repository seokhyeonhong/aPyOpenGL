from __future__ import annotations
from typing import Callable
from enum import Enum

from OpenGL.GL import *
import glm
import functools
import numpy as np

from . import core
from .material   import Material
# from .shader     import Shader
from .texture    import Texture, TextureLoader, TextureType
from .text       import FontTexture
from .motion     import Pose
from .mesh       import Mesh
from .model      import Model
from .obj        import Obj
from .const      import *
from .fbx        import FBX

def get_draw_func(render_func):
    if render_func is "phong":
        return functools.partial(Render.draw, pbr=False)
    elif render_func is "pbr":
        return functools.partial(Render.draw, pbr=True)
    else:
        raise Exception(f"Unknown render function: {render_func}")

class RenderMode(Enum):
    eDRAW   = 0
    eSHADOW = 1
    eTEXT   = 2

class RenderInfo:
    sky_color         = glm.vec4(0.8)
    cam_position      = glm.vec3(0.0)
    cam_projection    = glm.mat4(1.0)
    cam_view          = glm.mat4(1.0)
    light_vector      = glm.vec4(0.0)
    light_color       = glm.vec3(1.0)
    light_attenuation = glm.vec3(0.0)
    light_matrix      = glm.mat4(1.0)
    width             = 1920
    height            = 1080

""" Global rendering state and functions """
class Render:
    # rendering state
    render_mode  = RenderMode.eDRAW
    render_info  = RenderInfo()
    font_texture = None

    # shaders
    primitive_shader: core.Shader = None
    lbs_shader: core.Shader       = None
    text_shader: core.Shader      = None
    cubemap_shader: core.Shader   = None
    shadow_shader: core.Shader    = None
    equirect_shader: core.Shader  = None
    shaders: list[core.Shader]    = []

    # shadow map
    shadowmap_fbo        = None
    shadowmap_texture_id = None

    # irradiance map
    irradiance_map = None

    @staticmethod
    def initialize_shaders():
        Render.primitive_shader = core.Shader("vert.vs", "frag.fs")
        Render.lbs_shader       = core.Shader("lbs.vs", "frag.fs")
        Render.text_shader      = core.Shader("text.vs", "text.fs")
        Render.cubemap_shader   = core.Shader("cubemap.vs", "cubemap.fs")
        Render.shadow_shader    = core.Shader("shadow.vs", "shadow.fs")
        Render.equirect_shader  = core.Shader("equirect.vs", "equirect.fs")

        # shadow map
        Render.shadowmap_fbo, Render.shadowmap_texture_id = TextureLoader.generate_shadow_buffer()

        # irradiance map
        TextureLoader.load_irradiance_map(BACKGROUND_TEXTURE_FILE, Render.equirect_shader)

        # list of all shaders
        Render.shaders = [
            Render.primitive_shader,
            Render.lbs_shader,
            Render.shadow_shader,
            Render.text_shader,
            Render.cubemap_shader,
            Render.equirect_shader
        ]
    
    @staticmethod
    def sky_color():
        return Render.render_info.sky_color
    
    @staticmethod
    def set_render_mode(mode):
        if mode == RenderMode.eSHADOW:
            glBindFramebuffer(GL_FRAMEBUFFER, Render.shadowmap_fbo)
        else:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
        Render.render_mode = mode

    @staticmethod
    def cube(render_mode="pbr"):
        return RenderOptions(core.Cube(), Render.primitive_shader, get_draw_func(render_mode), Render.shadow_shader, Render.draw_shadow)

    @staticmethod
    def sphere(radius=0.5, stacks=16, sectors=16, render_mode="pbr"):
        return RenderOptions(core.Sphere(radius, stacks, sectors), Render.primitive_shader, get_draw_func(render_mode), Render.shadow_shader, Render.draw_shadow)
    
    @staticmethod
    def cone(radius=0.5, height=1, sectors=16, render_mode="pbr"):
        return RenderOptions(core.Cone(radius, height, sectors), Render.primitive_shader, get_draw_func(render_mode), Render.shadow_shader, Render.draw_shadow)

    @staticmethod
    def plane(width=1.0, height=1.0, render_mode="pbr"):
        return RenderOptions(core.Plane(width, height), Render.primitive_shader, get_draw_func(render_mode), Render.shadow_shader, Render.draw_shadow)
    
    @staticmethod
    def cylinder(radius=0.5, height=1, sectors=16, render_mode="pbr"):
        return RenderOptions(core.Cylinder(radius, height, sectors), Render.primitive_shader, get_draw_func(render_mode), Render.shadow_shader, Render.draw_shadow)

    @staticmethod
    def arrow(render_mode="pbr"):
        return RenderOptions(core.Arrow(), Render.primitive_shader, get_draw_func(render_mode), Render.shadow_shader, Render.draw_shadow)

    @staticmethod
    def pyramid(radius=0.5, height=1.0, sectors=4, render_mode="pbr"):
        return RenderOptions(core.Pyramid(radius, height, sectors), Render.primitive_shader, get_draw_func(render_mode), Render.shadow_shader, Render.draw_shadow)

    @staticmethod
    def heightmap(heightmap, render_mode="pbr"):
        return RenderOptions(heightmap.vao, Render.primitive_shader, get_draw_func(render_mode), Render.shadow_shader, Render.draw_shadow)
    
    @staticmethod
    def obj(filename, render_mode="pbr", scale=0.01):
        obj = Obj(filename, scale=scale)
        ro = RenderOptions(obj, Render.primitive_shader, get_draw_func(render_mode), Render.shadow_shader, Render.draw_shadow)
        ro.materials(obj.materials)
        return ro

    @staticmethod
    def mesh(mesh: Mesh, render_mode="pbr"):
        if mesh.use_skinning:
            ro = RenderOptions(mesh.mesh_gl.vao, Render.lbs_shader, get_draw_func(render_mode), Render.shadow_shader, Render.draw_shadow)
            ro.skinning(True).buffer_xforms(mesh.buffer)
        else:
            ro = RenderOptions(mesh.mesh_gl.vao, Render.primitive_shader, get_draw_func(render_mode), Render.shadow_shader, Render.draw_shadow)
        
        ro.materials(mesh.materials)
        return ro

    @staticmethod
    def model(model: Model, render_mode="pbr"):
        meshes = model.meshes
        rov = []
        for mesh in meshes:
            rov.append(Render.mesh(mesh, render_mode))

        return RenderOptionsVec(rov)
    
    @staticmethod
    def skeleton(model: Model, render_mode="pbr"):
        rov = []
        for idx, joint in enumerate(model.pose.get_skeleton().get_joints()[1:]):
            skeleton_xform = model.pose.get_skeleton_xforms()[idx]
            position = glm.vec3(skeleton_xform[:3, 3].ravel())
            orientation = glm.mat3(*skeleton_xform[:3, :3].T.ravel())
            bone_len = np.linalg.norm(joint.get_local_p())

            ro = Render.pyramid(radius=0.01, height=bone_len, render_mode=render_mode)
            ro.transform(position, orientation).albedo([0, 1, 0]).color_mode(True)
            rov.append(ro)
        return RenderOptionsVec(rov)

    @staticmethod
    def axis(render_mode="pbr"):
        fbx_axis = FBX(AXIS_MODEL_PATH, scale=0.01).model()
        rov = []
        for mesh in fbx_axis.meshes:
            ro = RenderOptions(mesh.mesh_gl.vao, Render.primitive_shader, get_draw_func(render_mode), None, None)
            ro.materials(mesh.materials)
            rov.append(ro)
        return RenderOptionsVec(rov)

    @staticmethod
    def text(t="", color=glm.vec3(0)):
        if Render.font_texture is None:
            Render.font_texture = FontTexture()

        res = RenderOptions(core.VAO(), Render.text_shader, functools.partial(Render.draw_text, on_screen=False))
        return res.text(str(t)).albedo(color)

    @staticmethod
    def text_on_screen(t="", color=glm.vec3(0)):
        if Render.font_texture is None:
            Render.font_texture = FontTexture()

        res = RenderOptions(core.VAO(), Render.text_shader, functools.partial(Render.draw_text, on_screen=True))
        return res.text(str(t)).albedo(color)

    @staticmethod
    def cubemap(dirname):
        ro = RenderOptions(core.Cubemap(), Render.cubemap_shader, Render.draw_cubemap)
        ro.cubemap(dirname)
        return ro

    @staticmethod
    def draw(option: RenderOptions, shader: core.Shader, pbr=True):
        if option is None or shader is None:
            return
        if Render.render_mode != RenderMode.eDRAW:
            return

        shader.use()

        # update shading mode
        shader.set_bool("uPBR", pbr)
        
        # update view
        if shader.is_view_updated is False:
            shader.set_mat4("uPV",                Render.render_info.cam_projection * Render.render_info.cam_view)
            shader.set_vec3("uViewPosition",      Render.render_info.cam_position)
            shader.set_vec4("uLight.vector",      Render.render_info.light_vector)
            shader.set_vec3("uLight.color",       Render.render_info.light_color)
            shader.set_vec3("uLight.attenuation", Render.render_info.light_attenuation)
            shader.set_mat4("uLightSpaceMatrix",  Render.render_info.light_matrix)
            shader.set_vec3("uSkyColor",          Render.render_info.sky_color)
            shader.is_view_updated = True

        # update model
        if option._use_skinning:
            if len(option._buffer_xforms) > MAX_JOINT_NUM:
                print(f"Joint number exceeds the limit: {len(option._buffer_xforms)} > {MAX_JOINT_NUM}")
                option._buffer_xforms = option._buffer_xforms[:MAX_JOINT_NUM]
                
            shader.set_mat4_array("uLbsJoints", option._buffer_xforms)
        else:
            T = glm.translate(glm.mat4(1.0), option._position)
            R = glm.mat4(option._orientation)
            S = glm.scale(glm.mat4(1.0), option._scale)
            transform = T * R * S
            shader.set_mat4("M", transform)

        # texture indexing
        if shader.is_texture_updated is False:
            shader.set_int("uIrradianceMap", 0)
            shader.set_int("uShadowMap", 1)
            shader.set_int_array("uTextures", [i + 2 for i in range(MAX_MATERIAL_TEXTURES)])
            shader.is_texture_updated = True
        
        # set irradiance map
        glActiveTexture(GL_TEXTURE0)
        irradiance_map = TextureLoader.load_irradiance_map(BACKGROUND_TEXTURE_FILE, Render.equirect_shader)
        glBindTexture(GL_TEXTURE_CUBE_MAP, irradiance_map.texture_id)
        shader.set_float("uIrradianceMapIntensity", option._background_intensity)

        # set shadow map
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, Render.shadowmap_texture_id)
        
        # remove all textures
        for i in range(MAX_MATERIAL_TEXTURES):
            glActiveTexture(GL_TEXTURE2 + i)
            glBindTexture(GL_TEXTURE_2D, 0)
        
        # material settings
        rgba = [glm.vec4(1)] * MAX_MATERIAL_NUM
        # attribs = [glm.vec3(0)] * MAX_MATERIAL_NUM
        texture_id = [glm.ivec3(-1)] * MAX_MATERIAL_NUM
        pbr_texture_id = [glm.ivec3(-1)] * MAX_MATERIAL_NUM

        def gl_set_texture(texture_id, count):
            if texture_id == 0 or count[0] >= MAX_MATERIAL_TEXTURES:
                return -1
            
            idx_on_shader = count[0]
            glActiveTexture(GL_TEXTURE2 + count[0])
            glBindTexture(GL_TEXTURE_2D, texture_id)
            count[0] += 1
            return idx_on_shader
        
        texture_count = [0] # use list to pass by reference
        for i in range(min(len(option._materials), MAX_MATERIAL_NUM)):
            material = option._materials[i]
            rgba[i] = glm.vec4(material.albedo, material.alpha)
            
            texture_id[i].x = gl_set_texture(material.albedo_map.texture_id, texture_count)
            texture_id[i].y = gl_set_texture(material.normal_map.texture_id, texture_count)
            texture_id[i].z = gl_set_texture(material.disp_map.texture_id, texture_count)

            if pbr:
                pbr_texture_id[i].x = gl_set_texture(material.metallic_map.texture_id, texture_count)
                pbr_texture_id[i].y = gl_set_texture(material.roughness_map.texture_id, texture_count)
                pbr_texture_id[i].z = gl_set_texture(material.ao_map.texture_id, texture_count)
        
        for i in range(min(len(option._materials), MAX_MATERIAL_NUM)):
            shader.set_vec4 (f"uMaterial[{i}].albedo",       rgba[i])
            shader.set_ivec3(f"uMaterial[{i}].textureID",    texture_id[i])
            shader.set_ivec3(f"uMaterial[{i}].pbrTextureID", pbr_texture_id[i])

            if pbr:
                shader.set_float(f"uMaterial[{i}].metallic",  option._materials[i].metallic)
                shader.set_float(f"uMaterial[{i}].roughness", option._materials[i].roughness)
                shader.set_float(f"uMaterial[{i}].ao",        option._materials[i].ao)
            else:
                shader.set_vec3 (f"uMaterial[{i}].diffuse",   option._materials[i].diffuse)
                shader.set_vec3 (f"uMaterial[{i}].specular",  option._materials[i].specular)
                shader.set_float(f"uMaterial[{i}].shininess", option._materials[i].shininess)

        shader.set_bool("uIsFloor",       option._is_floor)
        shader.set_vec3("uGridColor",     option._grid_color)
        shader.set_float("uGridWidth",    option._grid_width)
        shader.set_float("uGridInterval", option._grid_interval)

        shader.set_bool("uColorMode",  option._color_mode)
        shader.set_vec2("uvScale",     option._uv_repeat)
        shader.set_float("uDispScale", option._disp_scale)

        # final rendering
        glBindVertexArray(option._vao.id)
        glDrawElements(GL_TRIANGLES, len(option._vao.indices), GL_UNSIGNED_INT, None)

        # unbind vao
        glBindVertexArray(0)

    @staticmethod
    def draw_shadow(option: RenderOptions, shader: core.Shader):
        if option is None or shader is None:
            return
        if Render.render_mode != RenderMode.eSHADOW:
            return
        
        shader.use()

        shader.set_mat4("uLightSpaceMatrix", Render.render_info.light_matrix)

        if option._use_skinning:
            shader.set_bool(f"uIsSkinned", True)
            shader.set_mat4_array("uLbsJoints", option._buffer_xforms)
        else:
            shader.set_bool(f"uIsSkinned", False)
            T = glm.translate(glm.mat4(1.0), option._position)
            R = glm.mat4(option._orientation)
            S = glm.scale(glm.mat4(1.0), option._scale)
            shader.set_mat4("M", T * R * S)

        # final rendering
        glCullFace(GL_FRONT)
        glBindVertexArray(option._vao.id)
        glDrawElements(GL_TRIANGLES, len(option._vao.indices), GL_UNSIGNED_INT, None)
        glCullFace(GL_BACK)

        # unbind vao
        glBindVertexArray(0)

    @staticmethod
    def draw_text(option: RenderOptions, shader: core.Shader, on_screen=False):
        if option is None or shader is None:
            return
        if Render.render_mode != RenderMode.eTEXT:
            return
        

        if on_screen:
            glDisable(GL_DEPTH_TEST)
            x = option._position.x * (Render.render_info.width / 3840)
            y = option._position.y * (Render.render_info.width / 3840)
            scale = option._scale.x * (Render.render_info.width / 3840)
        else:
            x = 0
            y = 0
            scale = option._scale.x / TEXT_RESOLUTION

        # shader settings
        shader.use()
        
        if on_screen:
            PV = glm.ortho(0, Render.render_info.width, 0, Render.render_info.height)
            M  = glm.mat4(1.0)
        else:
            PV = Render.render_info.cam_projection * Render.render_info.cam_view
            M = glm.translate(glm.mat4(1.0), option._position)\
                * glm.mat4(option._orientation)\
                * glm.scale(glm.mat4(1.0), option._scale) # translation * rotation * scale

        shader.set_mat4("uPVM", PV * M)
        shader.set_int("uFontTexture", 0)
        shader.set_vec3("uTextColor", option._materials[0].albedo.xyz)

        glActiveTexture(GL_TEXTURE0)
        glBindVertexArray(Render.font_texture.vao)

        for c in option._text:
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
        
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindVertexArray(0)

        if on_screen:
            glEnable(GL_DEPTH_TEST)

    @staticmethod
    def draw_cubemap(option: RenderOptions, shader: core.Shader):
        if option is None or shader is None:
            return
        if Render.render_mode != RenderMode.eDRAW:
            return
        if option._cubemap.texture_id == 0:
            return
        
        shader.use()

        # update view
        P = Render.render_info.cam_projection
        V = glm.mat4(glm.mat3(Render.render_info.cam_view))
        shader.set_mat4("PV", P * V)

        # set textures
        shader.set_int("uSkybox", 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, option._cubemap.texture_id)

        # final rendering
        glBindVertexArray(option._vao.id)
        glDrawArrays(GL_TRIANGLES, 0, 36)

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
        Render.render_info.width             = width
        Render.render_info.height            = height

        for s in Render.shaders:
            s.is_view_updated = False
    
    @staticmethod
    def clear():
        Render.font_texture = None
        TextureLoader.clear()

""" Rendering options for a primitive (e.g. position, orientation, material, etc.) """
class RenderOptions:
    def __init__(
        self,
        vao: core.VAO,
        shader: core.Shader,
        draw_func: Callable[[RenderOptions, core.Shader], None],
        shadow_shader: core.Shader = None,
        shadow_func: Callable[[RenderOptions, core.Shader], None] = None,
    ):
        self._vao           = vao
        self._shader        = shader
        self._shadow_shader = shadow_shader

        # transformation
        self._position      = glm.vec3(0.0)
        self._orientation   = glm.mat3(1.0)
        self._scale         = glm.vec3(1.0)

        # joint
        self._use_skinning  = False
        self._buffer_xforms = []

        # material
        self._materials     = [Material()]
        self._cubemap       = Texture()
        self._uv_repeat     = glm.vec2(1.0)
        self._disp_scale    = 0.0001
        self._text          = ""
        self._color_mode    = False

        # grid and environment
        self._is_floor      = False
        self._grid_color    = glm.vec3(0.0)
        self._grid_width    = 1.0
        self._grid_interval = 1.0
        self._background_intensity = 1.0

        # visibility
        self._visible       = True
        self._draw_func     = draw_func
        self._shadow_func   = shadow_func

    def draw(self):
        if not self._visible:
            return
        
        if Render.render_mode == RenderMode.eSHADOW:
            if self._shadow_func is not None:
                self._shadow_func(self, self._shadow_shader)
        else:
            self._draw_func(self, self._shader)

    def position(self, position):
        self._position = glm.vec3(position)
        return self

    def orientation(self, orientation):
        self._orientation = glm.mat3(orientation)
        return self
    
    def transform(self, position, orientation):
        self._position = glm.vec3(position)
        self._orientation = glm.mat3(orientation)
        return self
    
    def scale(self, scale):
        self._scale = glm.vec3(scale)
        return self
    
    def albedo(self, color, material_id=0):
        if len(self._materials) == 0:
            self._materials.append(Material())
            material_id = 0
        
        if material_id < len(self._materials):
            self._materials[material_id].albedo = glm.vec3(color)
            self._materials[material_id].albedo_map.texture_id = 0

        return self
    
    def metallic(self, metallic, material_id=0):
        if len(self._materials) == 0:
            self._materials.append(Material())
            material_id = 0
        
        if material_id < len(self._materials):
            self._materials[material_id].metallic = metallic

        return self
    
    def roughness(self, roughness, material_id=0):
        if len(self._materials) == 0:
            self._materials.append(Material())
            material_id = 0
        
        if material_id < len(self._materials):
            self._materials[material_id].roughness = roughness

        return self

    def material(self, material, material_id=0):
        if len(self._materials) == 0:
            self._materials.append(Material())
            material_id = 0
        
        if material_id < len(self._materials):
            self._materials[material_id] = material

        return self
    
    def materials(self, materials):
        self._materials = materials
        return self
    
    def text_color(self, color):
        self._materials[0].albedo(glm.vec3(color))
        return self

    def texture(self, filename, texture_type=TextureType.eALBEDO, material_id=0):
        if len(self._materials) == 0:
            self._materials.append(Material())
            material_id = 0
        
        if material_id < len(self._materials):
            self._materials[material_id].set_texture(TextureLoader.load(filename), texture_type)

        return self
    
    def cubemap(self, dirname):
        self._cubemap = TextureLoader.load_cubemap(dirname)
        return self

    def floor(self, is_floor, line_width=1.0, line_interval=1.0, line_color=glm.vec3(0.0)):
        self._is_floor = is_floor
        self._grid_color = line_color
        self._grid_width = line_width
        self._grid_interval = line_interval
        return self
    
    def background(self, background_intensity):
        self._background_intensity = min(max(background_intensity, 0.0), 1.0)
        return self
    
    def skinning(self, use_skinning):
        self._use_skinning = use_skinning
        return self
    
    def buffer_xforms(self, buffer_xforms):
        self._buffer_xforms = buffer_xforms
        return self
    
    def uv_repeat(self, u, v=None):
        if v is None:
            self._uv_repeat = glm.vec2(u)
        else:
            self._uv_repeat = glm.vec2(u, v)
        return self

    def disp_scale(self, scale):
        self._disp_scale = scale
        return self
    
    def text(self, text):
        self._text = str(text)
        return self
    
    def alpha(self, alpha, material_id=0):
        if len(self._materials) == 0:
            self._materials.append(Material())
            material_id = 0

        if material_id < len(self._materials):
            self._materials[material_id].alpha(alpha)

        return self
    
    def color_mode(self, color_mode):
        self._color_mode = color_mode
        return self

    def switch_visible(self):
        self._visible = not self._visible
        return self
    
    def visible(self, visible):
        self._visible = visible
        return self

""" Multiple rendering options for a primitive """
class RenderOptionsVec:
    def __init__(self, options: list[RenderOptions]):
        self.options = options

    def draw(self):
        for option in self.options:
            option.draw()

    def switch_visible(self):
        for option in self.options:
            option.switch_visible()
        return self

    def visible(self, visible):
        for option in self.options:
            option.visible(visible)
        return self
    
    def background(self, background_intensity):
        for option in self.options:
            option.background(background_intensity)
        return self
    
    def buffer_xforms(self, model: Model):
        for i in range(len(self.options)):
            self.options[i].buffer_xforms(model.meshes[i].buffer)
        return self
    
    def position(self, position):
        for option in self.options:
            option.position(position)
        return self
    
    def position_of(self, index, position):
        self.options[index].position(position)
        return self

    def orientation(self, orientation):
        for option in self.options:
            option.orientation(orientation)
        return self
    
    def orientation_of(self, index, orientation):
        self.options[index].orientation(orientation)
        return self
    
    def transform(self, position, orientation):
        for option in self.options:
            option.transform(position, orientation)
        return self
    
    def transform_of(self, index, transform):
        self.options[index].transform(transform)
        return self
    
    def pose(self, pose: Pose):
        for idx, option in enumerate(self.options):
            xform = pose.get_skeleton_xforms()[idx]
            position = glm.vec3(*xform[:3, 3].ravel())
            orientation = glm.mat3(*xform[:3, :3].T.ravel())
            option.transform(position, orientation)
        return self
    
    # def set_all_positions(self, position):
    #     for option in self.options:
    #         option.set_position(position)
    #     return self

    # def set_all_scales(self, scale):
    #     for option in self.options:
    #         option.set_scale(scale)
    #     return self
    
    # def set_all_alphas(self, alpha):
    #     for option in self.options:
    #         for material in option.materials:
    #             material.set_alpha(alpha)
    #     return self
    
    # def set_all_color_modes(self, color_mode):
    #     for option in self.options:
    #         option.set_color_mode(color_mode)
    #     return self
    
    # def set_albedo(self, albedo, material_id=0):
    #     for option in self.options:
    #         option.set_albedo(albedo, material_id)
    #     return self
    
    # def set_albedo_of(self, albedo, model_id=0):
    #     self.options[model_id].set_albedo(albedo)
    #     return self
    
    # def set_alpha(self, alpha, material_id=0):
    #     for option in self.options:
    #         option.set_alpha(alpha, material_id)
    #     return self