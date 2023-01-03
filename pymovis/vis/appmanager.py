import glfw
from OpenGL.GL import *
import glm

from pymovis.vis.app import App
from pymovis.vis.render import Render, RenderMode
from pymovis.vis import glconst

class AppManager:
    """
    App cycle manager.
    """
    def __init__(
        self,
        app: App=None,
        width: int=1920,
        height: int=1080,
    ):
        self._app = app
        self._do_capture = False
        self.width = width
        self.height = height
        self.__initialize()

    def run(self, app: App):
        self._app = app
        self.__start_loop()

    def __initialize(self):
        # initialize glfw
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
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
    
    def __start_loop(self):
        if not isinstance(self._app, App):
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
        while glfw.window_should_close(self.window) == False:
            width, height = glfw.get_window_size(self.window)
            
            # sky color
            sky_color = Render.sky_color()
            glViewport(0, 0, width, height)
            glClearColor(sky_color.x, sky_color.y, sky_color.z, sky_color.a)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # update
            self._app.update()

            # update camera & light
            Render.update_render_view(self._app, width, height)

            # render shadow
            Render.set_render_mode(RenderMode.SHADOW, glconst.SHADOW_MAP_SIZE, glconst.SHADOW_MAP_SIZE)
            glViewport(0, 0, glconst.SHADOW_MAP_SIZE, glconst.SHADOW_MAP_SIZE)
            glClear(GL_DEPTH_BUFFER_BIT)
            self._app.render()

            # render scene
            Render.set_render_mode(RenderMode.PHONG, width, height)
            glViewport(0, 0, width, height)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self._app.render()

            # event
            glfw.poll_events()
            glfw.swap_buffers(self.window)

            # TODO: screen capture
        
        glfw.destroy_window(self.window)
        glfw.terminate()

    def on_key_down(self, window, key, scancode, action, mods):
        width, height = glfw.get_window_size(window)
        if key == glfw.KEY_F1 and action == glfw.PRESS:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        elif key == glfw.KEY_F2 and action == glfw.PRESS:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            
        self._app.key_callback(window, key, scancode, action, mods)

    def on_mouse_move(self, window, xpos, ypos):
        self._app.mouse_callback(window, xpos, ypos)

    def on_mouse_button_click(self, window, button, action, mods):
        self._app.mouse_button_callback(window, button, action, mods)

    def on_scroll(self, window, xoffset, yoffset):
        self._app.scroll_callback(window, xoffset, yoffset)

    def on_error(self, error, description):
        self._app.on_error(error, description)
    
    def on_resize(self, window, width, height):
        self.width = width
        self.height = height

        self._app.on_resize(window, width, height)