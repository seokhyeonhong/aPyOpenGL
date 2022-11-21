import os
import glm
from OpenGL.GL import *

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

def load(filename, shader_type):
    shader_code = load_code(filename)
    shader = glCreateShader(shader_type)
    
    glShaderSource(shader, shader_code)
    glCompileShader(shader)
    check_shader_compile_error(shader)
    
    return shader

def load_code(filename):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    shader_dir_path = os.path.join(curr_path, "shader")
    shader_path = os.path.join(shader_dir_path, filename)
    return open(shader_path, 'r').read()

class Shader:
    def __init__(self, vertex_path, fragment_path, geometry_path=None):
        self.vertex_shader   = load(vertex_path, GL_VERTEX_SHADER)
        self.fragment_shader = load(fragment_path, GL_FRAGMENT_SHADER)
        self.geometry_shader = load(geometry_path, GL_GEOMETRY_SHADER) if geometry_path != None else None
        self._build()
        
    def _build(self):
        self.program = glCreateProgram()
        glAttachShader(self.program, self.vertex_shader)
        glAttachShader(self.program, self.fragment_shader)
        if self.geometry_shader != None:
            glAttachShader(self.program, self.geometry_shader)
        
        glLinkProgram(self.program)
        check_program_link_error(self.program)

        glDeleteShader(self.vertex_shader)
        glDeleteShader(self.fragment_shader)
        if self.geometry_shader != None:
            glDeleteShader(self.geometry_shader)

        return self
        
    def use(self):
        glUseProgram(self.program)
    
    # set uniform variables in shader
    def set_int(self, name, value):   glUniform1i(glGetUniformLocation(self.program, name), value)
    def set_float(self, name, value): glUniform1f(glGetUniformLocation(self.program, name), value)
    def set_bool(self, name, value):  glUniform1i(glGetUniformLocation(self.program, name), value)
    def set_vec2(self, name, value):  glUniform2fv(glGetUniformLocation(self.program, name), 1, glm.value_ptr(value))
    def set_vec3(self, name, value):  glUniform3fv(glGetUniformLocation(self.program, name), 1, glm.value_ptr(value))
    def set_vec4(self, name, value):  glUniform4fv(glGetUniformLocation(self.program, name), 1, glm.value_ptr(value))
    def set_mat3(self, name, value):  glUniformMatrix3fv(glGetUniformLocation(self.program, name), 1, GL_FALSE, glm.value_ptr(value))
    def set_mat4(self, name, value):  glUniformMatrix4fv(glGetUniformLocation(self.program, name), 1, GL_FALSE, glm.value_ptr(value))