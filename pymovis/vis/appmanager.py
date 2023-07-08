import glfw
from OpenGL.GL import *

from .app    import App
from .render import Render, RenderMode
from .const  import SHADOW_MAP_SIZE

class AppManager:
    app = None

    @staticmethod
    def start(app: App):
        AppManager.initialize()
        AppManager.set_app(app)
        AppManager.render_loop()

    @staticmethod
    def initialize():
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)

    @staticmethod
    def set_app(app: App):
        AppManager.app = app

    @staticmethod
    def render_loop():
        if AppManager.app is None:
            raise Exception("AppManager.app is empty")
        
        # create window
        app = AppManager.app
        window = glfw.create_window(app.width, app.height, "pymovis", None, None)
        glfw.make_context_current(window)

        if not window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")
        
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        # callbacks
        glfw.set_framebuffer_size_callback(window, AppManager.on_resize)
        glfw.set_key_callback(window, AppManager.on_key_down)
        glfw.set_cursor_pos_callback(window, AppManager.on_mouse_move)
        glfw.set_mouse_button_callback(window, AppManager.on_mouse_button_click)
        glfw.set_scroll_callback(window, AppManager.on_scroll)
        glfw.set_error_callback(AppManager.on_error)
        
        # global OpenGL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_LINE_SMOOTH)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthFunc(GL_LESS)
        glCullFace(GL_BACK)
        # glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # intialize shaders
        Render.initialize_shaders()
        glViewport(0, 0, app.width, app.height)

        # start
        app.start()
        app.initialize_ui(window)

        # main loop
        glfw.set_time(0)
        while not glfw.window_should_close(window):
            # update window size
            width, height = glfw.get_window_size(window)
            glViewport(0, 0, width, height)

            # process inputs for ui
            app.process_inputs()
            
            # sky color
            sky_color = Render.sky_color()
            glClearColor(sky_color.x, sky_color.y, sky_color.z, sky_color.a)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # update
            app.update()

            # update camera & light
            Render.update_render_view(app, width, height)

            # render shadow
            Render.set_render_mode(RenderMode.eSHADOW)
            glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
            glClear(GL_DEPTH_BUFFER_BIT)
            app.render()

            # render scene
            Render.set_render_mode(RenderMode.eDRAW)
            glViewport(0, 0, width, height)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            app.render()

            # render background
            # Render.set_render_mode(RenderMode.eBACKGROUND)
            # glViewport(0, 0, BACKGROUND_MAP_SIZE, BACKGROUND_MAP_SIZE)
            # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            # app.render()

            # render text
            Render.set_render_mode(RenderMode.eTEXT)
            glViewport(0, 0, width, height)
            app.render_text()

            # render xray
            Render.set_render_mode(RenderMode.eDRAW)
            glClear(GL_DEPTH_BUFFER_BIT)
            app.render_xray()

            # late update
            app.late_update()

            # render ui
            app.render_ui()

            # event
            glfw.poll_events()
            glfw.swap_buffers(window)

        app.terminate_ui()
        glfw.destroy_window(window)
        glfw.terminate()
        app.terminate()

    @staticmethod
    def on_key_down(window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            Render.clear()
        elif key == glfw.KEY_F1 and action == glfw.PRESS:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        elif key == glfw.KEY_F2 and action == glfw.PRESS:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            
        AppManager.app.key_callback(window, key, scancode, action, mods)

    @staticmethod
    def on_mouse_move(window, xpos, ypos):
        AppManager.app.mouse_callback(window, xpos, ypos)

    @staticmethod
    def on_mouse_button_click(window, button, action, mods):
        AppManager.app.mouse_button_callback(window, button, action, mods)

    @staticmethod
    def on_scroll(window, xoffset, yoffset):
        AppManager.app.scroll_callback(window, xoffset, yoffset)

    @staticmethod
    def on_error(error, desc):
        AppManager.app.on_error(error, desc)
    
    @staticmethod
    def on_resize(window, width, height):
        AppManager.app.on_resize(window, width, height)