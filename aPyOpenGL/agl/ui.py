import imgui
from imgui.integrations.glfw import GlfwRenderer
import glfw
from OpenGL.GL import *

from .const import CONSOLAS_FONT_PATH

class UI:
    def __init__(self):
        self.window = None
        self.menu_to_items = {} # {menu_name: list[(item_name, func, key=None)]}
        self.key_to_func = {} # {key: list[func]}

    def initialize(self, window):
        self.window = window

        # imgui setup
        imgui.create_context()
        self.impl = GlfwRenderer(window, attach_callbacks=False)

        # IO - prevent creating a default window
        self.io = imgui.get_io()

        width, height = glfw.get_window_size(window)
        self.resize_font(width, height)
        self.font = self.io.fonts.add_font_from_file_ttf(CONSOLAS_FONT_PATH, 16)
        self.impl.refresh_font_texture()
    
    def resize_font(self, width, height):
        self.io.font_global_scale = min(width, height) / 750
        self.impl.refresh_font_texture()
        
    def process_inputs(self):
        self.impl.process_inputs()
        imgui.new_frame()
        imgui.push_font(self.font)

        if imgui.begin_main_menu_bar():
            # default menu
            if imgui.begin_menu("Menu", True):
                exit_clicked, exit_activated = imgui.menu_item("Exit", "ESC", False, True)
                if exit_clicked:
                    glfw.set_window_should_close(self.window, True)

                imgui.end_menu()

            # custom menus
            for menu_name, items in self.menu_to_items.items():
                imgui.separator()
                if imgui.begin_menu(menu_name, True):
                    max_len = 0
                    for item_name, _, _ in items:
                        max_len = max(max_len, imgui.calc_text_size(item_name)[0] + imgui.calc_text_size(item_name)[1] * 2 + imgui.get_style().item_spacing.x)
                    
                    for item_name, func, key in items:
                        clicked, activated = imgui.menu_item(item_name, "")
                        if key is not None:
                            text_key = glfw.get_key_name(key, 0)
                            if text_key is not None:
                                text_key = text_key.upper()
                            elif key is glfw.KEY_SPACE:
                                text_key = "Space"
                            elif key is glfw.KEY_LEFT:
                                text_key = "Left"
                            elif key is glfw.KEY_RIGHT:
                                text_key = "Right"

                            imgui.same_line(max_len)
                            imgui.text_disabled(text_key)

                        if clicked:
                            func()
                        
                    imgui.end_menu()
            imgui.end_main_menu_bar()
        
        # add slider to the glfw window
        # imgui.begin("Slider")
        # imgui.slider_float("float", 0, 1.0, 2.0, "%.3f")
        # imgui.end()


    def render(self, show_ui=True):
        imgui.pop_font()
        imgui.render()
        if show_ui:
            self.impl.render(imgui.get_draw_data())
        imgui.end_frame()
    
    def terminate(self):
        self.impl.shutdown()
    
    def add_menu(self, menu_name):
        self.menu_to_items[menu_name] = []
    
    def add_menu_item(self, menu_name, item_name, func, key=None):
        if self.menu_to_items.get(menu_name, None) is None:
            self.add_menu(menu_name)

        self.menu_to_items[menu_name].append([item_name, func, key])
        if key is not None:
            if self.key_to_func.get(key, None) is None:
                self.key_to_func[key] = []
            self.key_to_func[key].append(func)

    # def add_render_toggle(self, menu_name, item_name, render_option, key=None, activated=False):
    #     if self.menu_to_items.get(menu_name, None) is None:
    #         self.add_menu(menu_name)

    #     item = [render_option, key, activated]
    #     self.menu_to_items[menu_name][item_name] = item

    #     if key is not None:
    #         self.hotkey_to_render_options.setdefault(key, [])
    #         self.hotkey_to_render_options[key].append(render_option)
    
    def key_callback(self, window, key, scancode, action, mods):
        if not action == glfw.PRESS:
            return
        
        funcs = self.key_to_func.get(key, None)
        if funcs is not None:
            for func in funcs:
                func()