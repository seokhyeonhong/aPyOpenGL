import glfw
from OpenGL.GL import *

from pymovis.vis.app import App
from pymovis.vis.render import Render, RenderMode
from pymovis.vis.const import SHADOW_MAP_SIZE

class AppManager:
    def __init__(
        self,
        width: int  = 3960,
        height: int = 2160,
    ):
        self.do_capture = False
        self.width      = width
        self.height     = height
        self.initialize()

    def run(self, app: App):
        self.app = app
        self.app.width, self.app.height = self.width, self.height
        self.start_loop()

    def initialize(self):
        # initialize glfw
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(self.width, self.height, "Vis", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        glfw.set_framebuffer_size_callback(self.window, self.on_resize)
        glfw.set_key_callback(self.window, self.on_key_down)
        glfw.set_cursor_pos_callback(self.window, self.on_mouse_move)
        glfw.set_mouse_button_callback(self.window, self.on_mouse_button_click)
        glfw.set_scroll_callback(self.window, self.on_scroll)
        glfw.set_error_callback(self.on_error)

        # intialize shaders
        Render.initialize_shaders()
    
    def start_loop(self):
        if not isinstance(self.app, App):
            raise Exception("Invalid app type")
            
        # global OpenGL state
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        # glEnable(GL_FRAMEBUFFER_SRGB)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # glEnable(GL_LINE_SMOOTH)
        # glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # main loop
        glfw.set_time(0)
        while not glfw.window_should_close(self.window):
            width, height = glfw.get_window_size(self.window)
            glViewport(0, 0, width, height)
            
            # sky color
            sky_color = Render.sky_color()
            glClearColor(sky_color.x, sky_color.y, sky_color.z, sky_color.a)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # update
            self.app.update()

            # update camera & light
            Render.update_render_view(self.app, width, height)

            # render shadow
            Render.set_render_mode(RenderMode.eSHADOW, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
            glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
            glClear(GL_DEPTH_BUFFER_BIT)
            self.app.render()

            # render scene
            Render.set_render_mode(RenderMode.ePHONG, width, height)
            glViewport(0, 0, width, height)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.app.render()

            # late update
            self.app.late_update()

            # event
            glfw.poll_events()
            glfw.swap_buffers(self.window)

        glfw.destroy_window(self.window)
        glfw.terminate()
        self.app.terminate()

    def on_key_down(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            Render.clear()
        elif key == glfw.KEY_F1 and action == glfw.PRESS:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        elif key == glfw.KEY_F2 and action == glfw.PRESS:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            
        self.app.key_callback(window, key, scancode, action, mods)

    def on_mouse_move(self, window, xpos, ypos):
        self.app.mouse_callback(window, xpos, ypos)

    def on_mouse_button_click(self, window, button, action, mods):
        self.app.mouse_button_callback(window, button, action, mods)

    def on_scroll(self, window, xoffset, yoffset):
        self.app.scroll_callback(window, xoffset, yoffset)

    def on_error(self, error, description):
        self.app.on_error(error, description)
    
    def on_resize(self, window, width, height):
        self.width, self.height = width, height
        self.app.on_resize(window, width, height)