import glfw
from OpenGL.GL import *

from pymovis.vis.app import App
from pymovis.vis.render import Render, RenderMode
from pymovis.vis.const import SHADOW_MAP_SIZE
from pymovis.vis.ui import UI

class AppManager:
    def __init__(
        self,
        width:  int = 3840,
        height: int = 2160,
        maximize: bool = False,
    ):
        self.do_capture = False
        self.width      = width
        self.height     = height
        self.initialize(maximize)

    def run(self, app: App):
        self.app = app
        self.render_loop()

    def initialize(self, maximize: bool):
        # initialize glfw
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)

        self.window = glfw.create_window(self.width, self.height, "pymovis", None, None)
        glfw.make_context_current(self.window)
        
        if maximize:
            glfw.maximize_window(self.window)
            self.width, self.height = glfw.get_window_size(self.window)

        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # callbacks
        glfw.set_framebuffer_size_callback(self.window, self.on_resize)
        glfw.set_key_callback(self.window, self.on_key_down)
        glfw.set_cursor_pos_callback(self.window, self.on_mouse_move)
        glfw.set_mouse_button_callback(self.window, self.on_mouse_button_click)
        glfw.set_scroll_callback(self.window, self.on_scroll)
        glfw.set_error_callback(self.on_error)

        # intialize shaders
        Render.initialize_shaders()

    def render_loop(self):
        if not isinstance(self.app, App):
            raise Exception("Invalid app type")
        
        # initialize app window
        self.app.init_window(self.window)
        
        # global OpenGL state
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        # glEnable(GL_FRAMEBUFFER_SRGB)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_MULTISAMPLE)

        # glEnable(GL_LINE_SMOOTH)
        # glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # main loop
        glfw.set_time(0)
        while not glfw.window_should_close(self.window):
            # update window size
            width, height = glfw.get_window_size(self.window)
            glViewport(0, 0, width, height)

            # process inputs for ui
            self.app.process_inputs()
            
            # sky color
            sky_color = Render.sky_color()
            glClearColor(sky_color.x, sky_color.y, sky_color.z, sky_color.a)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # update
            self.app.update()

            # update camera & light
            Render.update_render_view(self.app, width, height)

            # render shadow
            Render.set_render_mode(RenderMode.eSHADOW)
            glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
            glClear(GL_DEPTH_BUFFER_BIT)
            self.app.render()

            # render scene
            Render.set_render_mode(RenderMode.eDRAW)
            glViewport(0, 0, width, height)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.app.render()

            # render background
            # Render.set_render_mode(RenderMode.eBACKGROUND)
            # glViewport(0, 0, BACKGROUND_MAP_SIZE, BACKGROUND_MAP_SIZE)
            # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            # self.app.render()

            # render text
            Render.set_render_mode(RenderMode.eTEXT)
            glViewport(0, 0, width, height)
            self.app.render_text()

            # late update
            self.app.late_update()

            # render ui
            self.app.render_ui()

            # event
            glfw.poll_events()
            glfw.swap_buffers(self.window)

        self.app.terminate_ui()
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

    def on_error(self, error, desc):
        self.app.on_error(error, desc)
    
    def on_resize(self, window, width, height):
        self.width, self.height = width, height
        self.app.on_resize(window, width, height)