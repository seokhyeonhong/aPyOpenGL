from OpenGL.GL import *

import glfw
import os
import datetime
import cv2
import numpy as np

from pymovis.motion.core import Motion

from pymovis.vis.camera import Camera
from pymovis.vis.light import DirectionalLight, PointLight
from pymovis.vis.render import Render

class App:
    def __init__(
        self,
        camera: Camera = Camera(),
        light: DirectionalLight = DirectionalLight(),
    ):
        self.camera = camera
        self.light = light
        
        self.width, self.height = 1920, 1080
        self.io = self.IO()

        self.capture_path = os.path.join("capture", str(datetime.date.today()))

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
        if key == glfw.KEY_V and action == glfw.PRESS:
            self.camera.switch_projection()
        elif key == glfw.KEY_F5 and action == glfw.PRESS:
            self.save_image(self.capture_screen())
        
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
    
    """ Capture functions """
    def capture_screen(self):
        viewport = glGetIntegerv(GL_VIEWPORT)
        x, y, *_ = viewport

        glReadBuffer(GL_FRONT)
        data = glReadPixels(x, y, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        pixels = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        pixels = np.flip(pixels, axis=0)
        image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        return image
    
    def save_image(self, image):
        image_dir = os.path.join(self.capture_path, "images")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        
        image_path = os.path.join(image_dir, datetime.datetime.now().strftime("%H-%M-%S") + ".png")
        cv2.imwrite(image_path, image)

class MotionApp(App):
    """ Class for motion capture visualization """
    def __init__(self, motion: Motion):
        super().__init__()
        self.motion = motion
        self.frame = 0
        self.prev_frame = -1
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
            if self.playing and self.frame == len(self.motion) - 1:
                self.frame = 0
                glfw.set_time(0)
        
        """ Move frames """
        if glfw.KEY_0 <= key <= glfw.KEY_9 and action == glfw.PRESS:
            self.frame = int(len(self.motion) * (key - glfw.KEY_0) * 0.1)
            glfw.set_time(self.frame / self.motion.fps)
            
        if not self.playing:
            if key == glfw.KEY_LEFT_BRACKET and action == glfw.PRESS:
                self.frame = max(self.frame - 1, 0)
            elif key == glfw.KEY_RIGHT_BRACKET and action == glfw.PRESS:
                self.frame = min(self.frame + 1, len(self.motion) - 1)
            if key == glfw.KEY_LEFT and action == glfw.PRESS:
                self.frame = max(self.frame - 10, 0)
            elif key == glfw.KEY_RIGHT and action == glfw.PRESS:
                self.frame = min(self.frame + 10, len(self.motion) - 1)
            glfw.set_time(self.frame / self.motion.fps)
        
        """ Render and capture options """
        if action == glfw.PRESS:
            if key == glfw.KEY_G:
                self.grid.switch_visible()
            elif key == glfw.KEY_A:
                self.axis.switch_visible()
            elif key == glfw.KEY_F6:
                if self.recording:
                    self.save_video(self.captures)
                    self.captures = []
                self.recording = not self.recording

    def update(self):
        """ Stop playing at the end of the motion"""
        if self.playing and self.frame == len(self.motion) - 1:
            self.playing = False

    def late_update(self):
        """ Rendering the current frame """
        Render.text(self.frame).draw()

        """ Recording """
        if self.recording:
            if self.prev_frame == self.frame and self.playing:
                return
                
            self.captures.append(super().capture_screen())
            self.prev_frame = self.frame

    def render(self):
        if self.playing:
            self.frame = min(int(glfw.get_time() * self.motion.fps), len(self.motion) - 1)
        else:
            glfw.set_time(self.frame / self.motion.fps)
        
        """ Render """
        self.grid.draw()
        self.motion.render_by_frame(self.frame)
        self.axis.draw()

    """ Capture functions """
    def save_video(self, captures):
        video_dir = os.path.join(self.capture_path, "videos")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        
        video_path = os.path.join(video_dir, datetime.datetime.now().strftime("%H-%M-%S") + ".mp4")
        fps = self.motion.fps
        height, width, _ = captures[0].shape
        
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for image in captures:
            video.write(image)
        video.release()

        glfw.set_time(self.frame / self.motion.fps)