from OpenGL.GL import *
from PIL import Image

import glfw
import os
import datetime

from pymovis.motion.core import Motion

from pymovis.vis.camera import Camera
from pymovis.vis.light import DirectionalLight, PointLight
from pymovis.vis.render import Render

class App:
    def __init__(
        self,
        camera: Camera = Camera(),
        light: DirectionalLight = DirectionalLight(),
        capture_path: str = "capture",
    ):
        self.camera = camera
        self.light = light
        self.width, self.height = 1920, 1080
        self.capture_path = os.path.join(capture_path, str(datetime.date.today()))
        if not os.path.exists(self.capture_path):
            os.makedirs(self.capture_path)
        self.io = self.IO()

    class IO:
        def __init__(self):
            self.last_mouse_x = 0
            self.last_mouse_y = 0
            self.mouse_middle_down = False
            self.mouse_left_down = False
    
    """ Override these methods to add custom rendering code. """
    def start(self):
        pass

    def update(self):
        pass

    def late_update(self):
        pass

    def render(self):
        pass
    
    def render_xray(self):
        pass
    
    """ Callback functions for glfw and camera control """
    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            Render.clear()
            
        # elif key == glfw.KEY_F1 and action == glfw.PRESS:
        #     Render.set_render_mode(RenderMode.PHONG)
        # elif key == glfw.KEY_F2 and action == glfw.PRESS:
        #     Render.set_render_mode(RenderMode.WIREFRAME)
        elif key == glfw.KEY_V and action == glfw.PRESS:
            self.camera.switch_projection()
        
    def mouse_callback(self, window, xpos, ypos):
        offset_x = xpos - self.io.last_mouse_x
        offset_y = self.io.last_mouse_y - ypos

        self.io.last_mouse_x = xpos
        self.io.last_mouse_y = ypos

        left_alt_pressed    = (glfw.get_key(window, glfw.KEY_LEFT_ALT) == glfw.PRESS)
        if left_alt_pressed and self.io.mouse_middle_down:
            self.camera.track(offset_x, offset_y)
        elif left_alt_pressed and self.io.mouse_left_down:
            self.camera.tumble(offset_x, offset_y)

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            self.io.mouse_left_down = True
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
            self.io.mouse_left_down = False
        if button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.PRESS:
            self.io.mouse_middle_down = True
        if button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.RELEASE:
            self.io.mouse_middle_down = False

    def scroll_callback(self, window, xoffset, yoffset):
        left_alt_pressed = (glfw.get_key(window, glfw.KEY_LEFT_ALT) == glfw.PRESS)
        if left_alt_pressed:
            self.camera.dolly(yoffset)
        else:
            self.camera.zoom(yoffset)

    def on_error(self, error, description):
        pass

    def on_resize(self, window, width, height):
        glViewport(0, 0, width, height)

class MotionApp(App):
    """ Class for motion capture visualization """
    def __init__(self, motion: Motion):
        super().__init__()
        self.motion = motion
        self.frame = 0
        self.playing = True
        self.recording = False
        self.captures = []

        self.grid = Render.plane().set_scale(50).set_uv_repeat(5).set_texture("grid.png")
        self.axis = Render.axis()
    
    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        """ Play / pause """
        if key == glfw.KEY_SPACE and action == glfw.PRESS:
            self.playing = not self.playing
        
        """ Move frames """
        if glfw.KEY_0 <= key <= glfw.KEY_9 and action == glfw.PRESS:
            self.frame = int(len(self.motion) * (key - glfw.KEY_0) * 0.1)
            glfw.set_time(self.frame / self.motion.fps)
            
        if not self.playing:
            if key == glfw.KEY_LEFT_BRACKET and action == glfw.PRESS:
                self.frame = max(self.frame - 1, 0)
            elif key == glfw.KEY_RIGHT_BRACKET and action == glfw.PRESS:
                self.frame = min(self.frame + 1, len(self.motion) - 1)
            if key == glfw.KEY_LEFT:
                self.frame = max(self.frame - 1, 0)
            elif key == glfw.KEY_RIGHT:
                self.frame = min(self.frame + 1, len(self.motion) - 1)
            glfw.set_time(self.frame / self.motion.fps)
        
        """ Render and capture options """
        if action == glfw.PRESS:
            if key == glfw.KEY_G:
                self.grid.switch_visible()
            elif key == glfw.KEY_A:
                self.axis.switch_visible()
            elif key == glfw.KEY_F5:
                self.save_image(self.capture_screen())
            elif key == glfw.KEY_F6:
                if self.recording:
                    self.save_video(self.captures)
                    self.captures = []
                self.recording = not self.recording

    def update(self):
        if self.recording:
            self.captures.append(self.capture_screen())

    def render(self):
        if self.playing:
            self.frame = min(int(glfw.get_time() * self.motion.fps), len(self.motion) - 1)
        else:
            glfw.set_time(self.frame / self.motion.fps)
        
        """ Render """
        self.grid.draw()
        self.motion.render_by_frame(self.frame)
        self.axis.draw()
        Render.text(self.frame).set_position(0, 0, 1).draw()

    """ Capture functions """
    def capture_screen(self):
        viewport = glGetIntegerv(GL_VIEWPORT)
        x = viewport[0]
        y = viewport[1]
        glReadBuffer(GL_FRONT)
        data = glReadPixels(x, y, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", (self.width, self.height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image
    
    def save_image(self, image):
        num_pngs = 0
        for file in os.listdir(self.capture_path):
            if file.endswith(".png"):
                num_pngs += 1
        image.save(os.path.join(self.capture_path, "{:04d}.png".format(num_pngs)))
    
    def save_video(self, captures):
        num_gifs = 0
        for file in os.listdir(self.capture_path):
            if file.endswith(".gif"):
                num_gifs += 1

        captures[0].save(os.path.join(self.capture_path, "{:04d}.gif".format(num_gifs)), save_all=True, append_images=captures[1:], optimize=False, duration=1000 / self.motion.fps, loop=0)