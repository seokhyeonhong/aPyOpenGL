import os
import imgui
import glfw
from imgui.integrations.glfw import GlfwRenderer
from OpenGL.GL import *

from pymovis.vis.const import CONSOLAS_FONT_PATH

class UI:
    def __init__(self):
        self.window = None
        self.menu_to_items = {} # {menu_name: {item_name: RenderOptions or RenderOptionsVec, hotkey, activated}}
        self.hotkey_to_render_options = {} # {hotkey: list of RenderOptions}

    def initialize(self, window):
        self.window = window

        # imgui setup
        imgui.create_context()
        self.impl = GlfwRenderer(window, attach_callbacks=False)

        # IO
        io = imgui.get_io()
        io.font_global_scale = 2.0
        self.font = io.fonts.add_font_from_file_ttf(CONSOLAS_FONT_PATH, 16)
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
                    for item_name in items.keys():
                        max_len = max(max_len, imgui.calc_text_size(item_name)[0] + imgui.calc_text_size(item_name)[1] * 2 + imgui.get_style().item_spacing.x)
                    
                    for item_name, (render_option, hotkey, activated) in items.items():
                        clicked, activated = imgui.checkbox(item_name, activated)
                        render_option.set_visible(activated)
                        items[item_name][2] = activated

                        if hotkey is not None:
                            hotkey = glfw.get_key_name(hotkey, 0).upper()
                            imgui.same_line(max_len)
                            imgui.text_disabled(hotkey)
                        
                    imgui.end_menu()
            imgui.end_main_menu_bar()

    def render(self):
        imgui.pop_font()
        imgui.render()
        self.impl.render(imgui.get_draw_data())
        imgui.end_frame()
    
    def terminate(self):
        self.impl.shutdown()
    
    def get_menu_height(self):
        return int(imgui.get_frame_height_with_spacing())
    
    def add_menu(self, menu_name):
        self.menu_to_items[menu_name] = {}

    def add_render_toggle(self, menu_name, item_name, render_option, key=None, activated=False):
        if self.menu_to_items.get(menu_name, None) is None:
            self.add_menu(menu_name)

        item = [render_option, key, activated]
        self.menu_to_items[menu_name][item_name] = item

        if key is not None:
            self.hotkey_to_render_options.setdefault(key, [])
            self.hotkey_to_render_options[key].append(render_option)
    
    def key_callback(self, window, key, scancode, action, mods):
        if not action == glfw.PRESS:
            return
        
        for hotkey, render_options in self.hotkey_to_render_options.items():
            if key == hotkey:
                for render_option in render_options:
                    render_option.switch_visible()