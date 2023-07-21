import os
import numpy as np
import glm
from OpenGL.GL import *

from ..const import AGL_PATH

class Shader:
    def __init__(self, vertex_path, fragment_path, geometry_path=None):
        self.vertex_shader    = load_shader(vertex_path, GL_VERTEX_SHADER)
        self.fragment_shader  = load_shader(fragment_path, GL_FRAGMENT_SHADER)
        self.geometry_shader  = load_shader(geometry_path, GL_GEOMETRY_SHADER) if geometry_path is not None else None
        self.build()

        self.name = os.path.splitext(os.path.basename(vertex_path))[0]
        self.is_view_updated    = False
        self.is_texture_updated = False
        
    def build(self):
        self.program = glCreateProgram()
        glAttachShader(self.program, self.vertex_shader)
        glAttachShader(self.program, self.fragment_shader)
        if self.geometry_shader is not None:
            glAttachShader(self.program, self.geometry_shader)
        
        glLinkProgram(self.program)
        check_program_link_error(self.program)

        glDeleteShader(self.vertex_shader)
        glDeleteShader(self.fragment_shader)
        if self.geometry_shader is not None:
            glDeleteShader(self.geometry_shader)
        
    def use(self):
        glUseProgram(self.program)
    
    """ Set uniform variables in shader """
    def set_int(self, name, value):   glUniform1i(glGetUniformLocation(self.program, name), value)
    def set_float(self, name, value): glUniform1f(glGetUniformLocation(self.program, name), value)
    def set_bool(self, name, value):  glUniform1i(glGetUniformLocation(self.program, name), value)
    def set_vec2(self, name, value):  glUniform2fv(glGetUniformLocation(self.program, name), 1, glm.value_ptr(value))
    def set_vec3(self, name, value):  glUniform3fv(glGetUniformLocation(self.program, name), 1, glm.value_ptr(value))
    def set_vec4(self, name, value):  glUniform4fv(glGetUniformLocation(self.program, name), 1, glm.value_ptr(value))
    def set_ivec3(self, name, value): glUniform3iv(glGetUniformLocation(self.program, name), 1, glm.value_ptr(value))
    def set_ivec4(self, name, value): glUniform4iv(glGetUniformLocation(self.program, name), 1, glm.value_ptr(value))
    def set_mat3(self, name, value):  glUniformMatrix3fv(glGetUniformLocation(self.program, name), 1, GL_FALSE, glm.value_ptr(value))
    def set_mat4(self, name, value):  glUniformMatrix4fv(glGetUniformLocation(self.program, name), 1, GL_FALSE, glm.value_ptr(value))

    # TODO: Impelement these functions
    # def set_struct(self, name, value, struct_name): glUniform1fv(glGetUniformLocation(self.program, name), f"{struct_name}.Type", value)
    # def set_multiple_float(self, name, value): glUniform1fv(glGetUniformLocation(self.program, name), len(value), value)
    # def set_multiple_vec3(self, name, value):  glUniform3fv(glGetUniformLocation(self.program, name), len(value), glm.value_ptr(value))
    # def set_multiple_vec4(self, name, value):  glUniform4fv(glGetUniformLocation(self.program, name), len(value), glm.value_ptr(value))
    # def set_multiple_ivec3(self, name, value): glUniform3iv(glGetUniformLocation(self.program, name), len(value), glm.value_ptr(value))
    # def set_multiple_ivec4(self, name, value): glUniform4iv(glGetUniformLocation(self.program, name), len(value), glm.value_ptr(value))

    def set_multiple_int(self, name, values): glUniform1iv(glGetUniformLocation(self.program, name), len(values), values)
    def set_multiple_float(self, name, values): glUniform1fv(glGetUniformLocation(self.program, name), len(values), values)

    def set_multiple_mat4(self, name, values): glUniformMatrix4fv(glGetUniformLocation(self.program, name), len(values), GL_FALSE, self._glm_values_to_ptr(values))

    def _glm_values_to_ptr(self, values):
        # transpose because glm is column-major while numpy is row-major
        float_array = np.concatenate([np.asarray(value, dtype=np.float32).transpose().flatten() for value in values])
        return float_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

def check_shader_compile_error(handle):
    success = glGetShaderiv(handle, GL_COMPILE_STATUS)
    if not success:
        info_log = glGetShaderInfoLog(handle)
        raise Exception("Shader compilation error: " + info_log.decode("utf-8"))

def check_program_link_error(handle):
    success = glGetProgramiv(handle, GL_LINK_STATUS)
    if not success:
        info_log = glGetProgramInfoLog(handle)
        raise Exception("Shader program linking error: " + info_log.decode("utf-8"))

def load_shader(filename, shader_type):
    shader_code = load_code(filename)
    shader = glCreateShader(shader_type)
    
    glShaderSource(shader, shader_code)
    glCompileShader(shader)
    check_shader_compile_error(shader)
    
    return shader

def load_code(filename):
    shader_dir_path = os.path.join(AGL_PATH, "shader")
    shader_path = os.path.join(shader_dir_path, filename)
    with open(shader_path, "r") as f:
        code = f.read()
    return code