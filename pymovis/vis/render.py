from __future__ import annotations
from enum import Enum
import functools

from OpenGL.GL import *
import glm

from pymovis.vis.primitives import *
from pymovis.vis.material import Material
from pymovis.vis.shader import Shader
from pymovis.vis.primitives import Cube
from pymovis.vis.texture import Texture, TextureType, TextureLoader
from pymovis.vis.text import FontTexture
from pymovis.vis.mesh import Mesh
from pymovis.vis.model import Model
from pymovis.vis.const import SHADOW_MAP_SIZE, TEXT_RESOLUTION, MAX_MATERIAL_NUM, MAX_MATERIAL_TEXTURES, MAX_JOINT_NUM

class RenderMode(Enum):
    ePHONG  = 0
    eSHADOW = 1

class RenderInfo:
    sky_color         = glm.vec4(1.0)
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
    render_mode      = RenderMode.ePHONG
    render_info      = RenderInfo()
    primitive_meshes = {}
    font_texture     = None

    @staticmethod
    def initialize_shaders():
        Render.primitive_shader = Shader("phong.vs",   "phong.fs")
        Render.lbs_shader       = Shader("lbs.vs",     "phong.fs")
        Render.shadow_shader    = Shader("shadow.vs",  "shadow.fs")
        Render.text_shader      = Shader("text.vs",    "text.fs")
        Render.cubemap_shader   = Shader("cubemap.vs", "cubemap.fs")
        Render.shaders = [Render.primitive_shader, Render.lbs_shader, Render.shadow_shader, Render.text_shader, Render.cubemap_shader]
        Render.generate_shadow_buffer()
    
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

        Render.depth_map = depth_map
        Render.depth_map_fbo = depth_map_fbo

    @staticmethod
    def sky_color():
        return Render.render_info.sky_color
    
    @staticmethod
    def set_render_mode(mode):
        if mode == RenderMode.eSHADOW:
            glBindFramebuffer(GL_FRAMEBUFFER, Render.depth_map_fbo)
        else:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
        Render.render_mode = mode

    @staticmethod
    def render_options(p):
        if Render.render_mode == RenderMode.eSHADOW:
            return RenderOptions(p, Render.shadow_shader, Render.draw_shadow)
        else:
            return RenderOptions(p, Render.primitive_shader, Render.draw_phong)

    @staticmethod
    def cube():
        return RenderOptions(Cube(), Render.primitive_shader, Render.draw_phong, Render.shadow_shader, Render.draw_shadow)

    @staticmethod
    def sphere(radius=0.5, stacks=16, sectors=16):
        return RenderOptions(Sphere(radius, stacks, sectors), Render.primitive_shader, Render.draw_phong, Render.shadow_shader, Render.draw_shadow)
    
    @staticmethod
    def cone(radius=0.5, height=1, sectors=16):
        return RenderOptions(Cone(radius, height, sectors), Render.primitive_shader, Render.draw_phong, Render.shadow_shader, Render.draw_shadow)

    @staticmethod
    def plane():
        return RenderOptions(Plane(), Render.primitive_shader, Render.draw_phong, Render.shadow_shader, Render.draw_shadow)
    
    @staticmethod
    def grid(size_x=1.0, size_z=1.0):
        return RenderOptions(Plane(), Render.primitive_shader, Render.draw_phong, Render.shadow_shader, Render.draw_shadow).set_floor(True).set_grid_size(size_x, size_z)

    @staticmethod
    def cylinder(radius=0.5, height=1, sectors=16):
        return RenderOptions(Cylinder(radius, height, sectors), Render.primitive_shader, Render.draw_phong, Render.shadow_shader, Render.draw_shadow)

    @staticmethod
    def arrow():
        return RenderOptions(Arrow(), Render.primitive_shader, Render.draw_phong, Render.shadow_shader, Render.draw_shadow)

    @staticmethod
    def pyramid(radius=0.5, height=1.0, sectors=4):
        return RenderOptions(Pyramid(radius, height, sectors), Render.primitive_shader, Render.draw_phong, Render.shadow_shader, Render.draw_shadow)

    @staticmethod
    def vao(vao):
        return RenderOptions(vao, Render.primitive_shader, Render.draw_phong, Render.shadow_shader, Render.draw_shadow)
    
    @staticmethod
    def heightmap(heightmap):
        return Render.vao(heightmap.vao)

    @staticmethod
    def mesh(mesh: Mesh):
        if mesh.use_skinning:
            ro = RenderOptions(mesh.mesh_gl.vao, Render.lbs_shader, Render.draw_phong, Render.shadow_shader, Render.draw_shadow)
            ro.set_skinning(True).set_buffer_transforms(mesh.buffer)
        else:
            ro = RenderOptions(mesh.mesh_gl.vao, Render.primitive_shader, Render.draw_phong, Render.shadow_shader, Render.draw_shadow)
        
        ro.set_materials(mesh.materials)
        return ro

    @staticmethod
    def model(model: Model):
        meshes = model.meshes
        rov = []
        for mesh in meshes:
            rov.append(Render.mesh(mesh))

        return RenderOptionsVec(rov)

    @staticmethod
    def axis():
        R_x = glm.rotate(glm.mat4(1.0), glm.radians(-90), glm.vec3(0, 0, 1))
        arrow_x = RenderOptions(Arrow(), Render.primitive_shader, Render.draw_phong).set_orientation(R_x).set_albedo(glm.vec3(1, 0, 0)).set_color_mode(True)
        
        arrow_y = RenderOptions(Arrow(), Render.primitive_shader, Render.draw_phong).set_albedo(glm.vec3(0, 1, 0)).set_color_mode(True)

        R_z = glm.rotate(glm.mat4(1.0), glm.radians(90), glm.vec3(1, 0, 0))
        arrow_z = RenderOptions(Arrow(), Render.primitive_shader, Render.draw_phong).set_orientation(R_z).set_albedo(glm.vec3(0, 0, 1)).set_color_mode(True)
        return RenderOptionsVec([arrow_x, arrow_y, arrow_z])

    @staticmethod
    def text(t="", color=glm.vec3(0)):
        if Render.font_texture is None:
            Render.font_texture = FontTexture()

        res = RenderOptions(VAO(), Render.text_shader, functools.partial(Render.draw_text, on_screen=False))
        return res.set_text(str(t)).set_albedo(color)

    @staticmethod
    def text_on_screen(t="", color=glm.vec3(0)):
        if Render.font_texture is None:
            Render.font_texture = FontTexture()

        res = RenderOptions(VAO(), Render.text_shader, functools.partial(Render.draw_text, on_screen=True))
        return res.set_text(str(t)).set_albedo(glm.vec3(0))

    # @staticmethod
    # def cubemap(dirname, scale=100):
    #     ro = RenderOptions(Cubemap(scale=scale), Render.cubemap_shader, Render.draw_cubemap)
    #     ro.set_cubemap(dirname)
    #     return ro

    @staticmethod
    def draw_phong(option: RenderOptions, shader: Shader):
        if option is None or shader is None:
            return
        if Render.render_mode == RenderMode.eSHADOW:
            return

        shader.use()
        
        # update view
        if shader.is_view_updated is False:
            shader.set_mat4("uPV",                Render.render_info.cam_projection * Render.render_info.cam_view)
            shader.set_vec3("uViewPosition",      Render.render_info.cam_position)
            shader.set_vec4("uLight.vector",      Render.render_info.light_vector)
            shader.set_vec3("uLight.color",       Render.render_info.light_color)
            shader.set_vec3("uLight.attenuation", Render.render_info.light_attenuation)
            shader.set_mat4("uLightSpaceMatrix",  Render.render_info.light_matrix)
            shader.is_view_updated = True

        # update model
        if option.use_skinning:
            if len(option.buffer_transforms) > MAX_JOINT_NUM:
                print(f"Joint number exceeds the limit: {len(option.buffer_transforms)} > {MAX_JOINT_NUM}")
                option.buffer_transforms = option.buffer_transforms[:MAX_JOINT_NUM]
                
            for idx, buffer in enumerate(option.buffer_transforms):
                shader.set_mat4(f"uLbsJoints[{idx}]", buffer)
        else:
            T = glm.translate(glm.mat4(1.0), option.position)
            R = glm.mat4(option.orientation)
            S = glm.scale(glm.mat4(1.0), option.scale)
            transform = T * R * S
            shader.set_mat4("M", transform)

        # texture indexing
        if shader.is_texture_updated is False:
            shader.set_int("uShadowMap", 1)
            for i in range(len(option.materials)):
                shader.set_int(f"uTextures[{i}]", i + 2)
            shader.is_texture_updated = True

        # TODO: fix this
        # set environment map 
        # shader.set_int("uMaterial.albedoMap", 0)
        # if option.texture_id is not None:
        #     shader.set_int("uMaterial.id", 0)
        #     glActiveTexture(GL_TEXTURE0)
        #     glBindTexture(GL_TEXTURE_2D, option.texture_id)
        # else:
        #     shader.set_int("uMaterial.id", -1)
        
        # set shadow map
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, Render.depth_map)
        
        # remove all textures
        for i in range(MAX_MATERIAL_TEXTURES):
            glActiveTexture(GL_TEXTURE2 + i)
            glBindTexture(GL_TEXTURE_2D, 0)
        
        # material settings
        rgba = [glm.vec4(1)] * MAX_MATERIAL_NUM
        # attribs = [glm.vec3(0)] * MAX_MATERIAL_NUM
        texture_id = [glm.ivec4(-1)] * MAX_MATERIAL_NUM

        def gl_set_texture(texture_id, count):
            if texture_id == 0 or count >= MAX_MATERIAL_TEXTURES:
                return -1
            
            idx_on_shader = count
            glActiveTexture(GL_TEXTURE2 + count)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            count += 1
            return idx_on_shader
        
        texture_count = 0
        for i in range(len(option.materials)):
            if i >= MAX_MATERIAL_NUM:
                break

            material = option.materials[i]
            rgba[i] = glm.vec4(material.albedo, material.alpha)

            texture_id[i].x = gl_set_texture(material.albedo_map.texture_id, texture_count)

        for i in range(len(option.materials)):
            shader.set_vec4(f"uMaterial[{i}].albedo", rgba[i])
            shader.set_vec3(f"uMaterial[{i}].diffuse", option.materials[i].diffuse)
            shader.set_vec3(f"uMaterial[{i}].specular", option.materials[i].specular)
            shader.set_float(f"uMaterial[{i}].shininess", option.materials[i].shininess)
            shader.set_ivec4(f"uMaterial[{i}].textureID", texture_id[i])

        shader.set_bool("uIsFloor", option.is_floor)
        shader.set_vec2("uGridSize", option.grid_size)
        for i in range(2):
            shader.set_vec3(f"uGridColors[{i}]", option.grid_colors[i])

        shader.set_bool("uColorMode", option.color_mode)
        shader.set_vec2("uvScale",    option.uv_repeat)

        # final rendering
        glBindVertexArray(option.vao.id)
        glDrawElements(GL_TRIANGLES, len(option.vao.indices), GL_UNSIGNED_INT, None)

        # unbind vao
        glBindVertexArray(0)
    
    @staticmethod
    def draw_shadow(option: RenderOptions, shader: Shader):
        if option is None or shader is None:
            return
        if Render.render_mode != RenderMode.eSHADOW:
            return
        
        shader.use()

        shader.set_mat4("uLightSpaceMatrix", Render.render_info.light_matrix)

        if option.use_skinning:
            shader.set_bool(f"uIsSkinned", True)
            for idx, buffer in enumerate(option.buffer_transforms):
                shader.set_mat4(f"uLbsJoints[{idx}]", buffer)
        else:
            shader.set_bool(f"uIsSkinned", False)
            T = glm.translate(glm.mat4(1.0), option.position)
            R = glm.mat4(option.orientation)
            S = glm.scale(glm.mat4(1.0), option.scale)
            shader.set_mat4("M", T * R * S)

        # final rendering
        glCullFace(GL_FRONT)
        glBindVertexArray(option.vao.id)
        glDrawElements(GL_TRIANGLES, len(option.vao.indices), GL_UNSIGNED_INT, None)
        glCullFace(GL_BACK)

        # unbind vao
        glBindVertexArray(0)

    @staticmethod
    def draw_text(option: RenderOptions, shader: Shader, on_screen=False):
        if option is None or shader is None:
            return
        
        if on_screen:
            x = option.position.x
            y = option.position.y
            scale = option.scale.x
        else:
            x = 0
            y = 0
            scale = option.scale.x / TEXT_RESOLUTION

        # shader settings
        shader.use()
        
        if on_screen:
            PV = glm.ortho(0, Render.render_info.width, 0, Render.render_info.height)
            M  = glm.mat4(1.0)
        else:
            PV = Render.render_info.cam_projection * Render.render_info.cam_view
            M = glm.translate(glm.mat4(1.0), option.position)\
                * glm.mat4(option.orientation)\
                * glm.scale(glm.mat4(1.0), option.scale) # translation * rotation * scale

        shader.set_mat4("uPVM", PV * M)
        shader.set_int("uFontTexture", 0)
        shader.set_vec3("uTextColor", option.materials[0].albedo.xyz)

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

    # @staticmethod
    # def draw_cubemap(option: RenderOptions, shader: Shader):
    #     if option is None or shader is None:
    #         return
    #     if Render.render_mode == RenderMode.eSHADOW:
    #         return
        
    #     # adjust depth settings for optimized rendering
    #     glDepthFunc(GL_LEQUAL)

    #     shader.use()

    #     # update view
    #     P = Render.render_info.cam_projection
    #     V = glm.mat4(glm.mat3(Render.render_info.cam_view))
    #     shader.set_mat4("PV", P * V)

    #     # set textures
    #     shader.set_int("uSkybox", 0)
    #     glActiveTexture(GL_TEXTURE0)
    #     glBindTexture(GL_TEXTURE_CUBE_MAP, option.cubemap_id)

    #     # final rendering
    #     glBindVertexArray(option.vao.id)
    #     glDrawArrays(GL_TRIANGLES, 0, 36)

    #     # restore depth settings
    #     glDepthFunc(GL_LESS)

    #     # unbind vao
    #     glBindVertexArray(0)


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
        Render.primitive_meshes.clear()
        Render.font_texture = None
        TextureLoader.clear()

""" Rendering options for a primitive (e.g. position, orientation, material, etc.) """
class RenderOptions:
    def __init__(
        self,
        vao: VAO,
        shader,
        draw_func,
        shadow_shader=None,
        shadow_func=None,
    ):
        self.vao           = vao
        self.shader        = shader
        self.shadow_shader = shadow_shader

        # transformation
        self.position      = glm.vec3(0.0)
        self.orientation   = glm.mat3(1.0)
        self.scale         = glm.vec3(1.0)

        # joint
        self.use_skinning  = False
        self.buffer_transforms = []

        # material
        self.materials     = [Material()]
        self.uv_repeat     = glm.vec2(1.0)
        self.text          = ""
        self.color_mode    = False
        self.is_floor      = False
        self.grid_size     = glm.vec2(1.0)
        self.grid_colors   = [glm.vec3(0.0), glm.vec3(1.0)]

        # visibility
        self.visible       = True
        self.draw_func     = draw_func
        self.shadow_func   = shadow_func

    def draw(self):
        if not self.visible:
            return
        
        if Render.render_mode == RenderMode.eSHADOW:
            if self.shadow_func is not None:
                self.shadow_func(self, self.shadow_shader)
        else:
            self.draw_func(self, self.shader)

    def set_position(self, x, y=None, z=None):
        if y is None and z is None:
            self.position = glm.vec3(x)
        elif y is not None and z is not None:
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

    def set_albedo(self, color, material_id=0):
        if len(self.materials) == 0:
            self.materials.append(Material())
            material_id = 0
        
        if material_id < len(self.materials):
            self.materials[material_id].albedo = glm.vec3(color)
            self.materials[material_id].albedo_map.texture_id = 0

        return self
    
    def set_material(self, material, material_id=0):
        if len(self.materials) == 0:
            self.materials.append(Material())
            material_id = 0
        
        if material_id < len(self.materials):
            self.materials[material_id] = material

        return self
    
    def set_materials(self, materials):
        self.materials = materials
        return self
    
    def set_text_color(self, color):
        self.materials[0].set_albedo(glm.vec3(color))
        return self

    def set_texture(self, filename, texture_type=TextureType.eALBEDO, material_id=0):
        if len(self.materials) == 0:
            self.materials.append(Material())
            material_id = 0
        
        if material_id < len(self.materials):
            self.materials[material_id].set_texture(TextureLoader.load(filename), texture_type)

        return self
    
    # def set_cubemap(self, dirname):
    #     self.material.set_texture(TextureLoader.load_cubemap(dirname), TextureType.eCUBEMAP)
    #     return self

    def set_floor(self, is_floor):
        self.is_floor = is_floor
        return self
    
    def set_grid_size(self, size_x=1.0, size_z=1.0):
        self.grid_size = glm.vec2(size_x, size_z)
        return self
    
    def set_grid_color(self, color1=glm.vec3(0.0), color2=glm.vec3(1.0)):
        self.grid_colors = [glm.vec3(color1), glm.vec3(color2)]
        return self

    def set_skinning(self, use_skinning):
        self.use_skinning = use_skinning
        return self
    
    def set_buffer_transforms(self, buffer_transforms):
        self.buffer_transforms = buffer_transforms
        return self
    
    def set_uv_repeat(self, u, v=None):
        if v is None:
            self.uv_repeat = glm.vec2(u)
        else:
            self.uv_repeat = glm.vec2(u, v)
        return self
    
    def set_text(self, text):
        self.text = str(text)
        return self
    
    def set_alpha(self, alpha, material_id=0):
        if len(self.materials) == 0:
            self.materials.append(Material())
            material_id = 0

        if material_id < len(self.materials):
            self.materials[material_id].set_alpha(alpha)

        return self
    
    def set_color_mode(self, color_mode):
        self.color_mode = color_mode
        return self

    def switch_visible(self):
        self.visible = not self.visible
        return self
    
    def set_visible(self, visible):
        self.visible = visible
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
    
    def set_all_alphas(self, alpha):
        for option in self.options:
            for material in option.materials:
                material.set_alpha(alpha)
        return self
    
    def set_albedo(self, albedo, material_id=0):
        for option in self.options:
            option.set_albedo(albedo, material_id)
        return self
    
    def set_alpha(self, alpha, material_id=0):
        for option in self.options:
            option.set_alpha(alpha, material_id)
        return self

    def set_visible(self, visible):
        for option in self.options:
            option.set_visible(visible)
        return self