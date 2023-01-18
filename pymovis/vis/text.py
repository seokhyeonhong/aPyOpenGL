import os, sys
import freetype as ft

from OpenGL.GL import *
import glm

from pymovis.vis import glconst


class FontTexture:
    def __init__(self, font_filename="consola.ttf"):
        class Character:
            def __init__(self, texture_id: int, size: glm.ivec2, bearing: glm.ivec2, advance: int):
                self.texture_id = texture_id
                self.size = size
                self.bearing = bearing
                self.advance = advance

        self.character_map = {}

        face = ft.Face(self.get_font_path(font_filename))
        face.set_pixel_sizes(0, glconst.TEXT_RESOLUTION)

        # disable byte-alignment restriction
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        ## load 128 ASCII characters
        for c in range(128):
            char = chr(c)
            face.load_char(char, ft.FT_LOAD_RENDER)

            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RED,
                face.glyph.bitmap.width,
                face.glyph.bitmap.rows,
                0,
                GL_RED,
                GL_UNSIGNED_BYTE,
                face.glyph.bitmap.buffer
            )

            # set texture options
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            # store character for later use
            character = Character(
                texture,
                glm.ivec2(face.glyph.bitmap.width, face.glyph.bitmap.rows),
                glm.ivec2(face.glyph.bitmap_left, face.glyph.bitmap_top),
                face.glyph.advance.x
            )
            self.character_map[char] = character
        
        glBindTexture(GL_TEXTURE_2D, 0)

        # VAO and VBO for text quads
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, sys.getsizeof(float) * 6 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
    
    def get_font_path(self, font_filename):
        curr_path = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(curr_path, "font", font_filename)
        return font_path
    
    def character(self, c):
        return self.character_map[c]
