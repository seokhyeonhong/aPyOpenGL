from OpenGL.GL import *

import glfw
import os
import time
import datetime
import cv2
import numpy as np
import glm

from pymovis.motion.core import Motion

from pymovis.vis.camera import Camera
from pymovis.vis.light import Light, DirectionalLight, PointLight
from pymovis.vis.render import Render
from pymovis.vis.model import Model
from pymovis.vis.const import LAFAN_BVH_TO_FBX

""" Base class for all applications """
class App:
    def __init__(
        self,
        camera = Camera(),
        light  = DirectionalLight(),
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
    
    """ Override these methods to add custom rendering code """
    def start(self):
        pass

    def update(self):
        pass

    def late_update(self):
        pass

    def render(self):
        pass
    
    def terminate(self):
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

""" Class for general animation with a fixed number of frames """
class AnimApp(App):
    def __init__(self, total_frames, fps=30):
        super().__init__()
        self.total_frames = int(total_frames)
        self.fps = fps

        self.frame = 0
        self.playing = True
        self.recording = False
        self.record_start_time = 0
        self.record_end_time = 0
        self.captures = []

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        # Space: play/pause
        if key == glfw.KEY_SPACE and action == glfw.PRESS:
            self.playing = not self.playing
        
        # Replay when the end of the animation is reached
        if self.playing and self.frame == self.total_frames - 1:
            self.frame = 0
            glfw.set_time(0)

        # 0~9, []: frame control
        if glfw.KEY_0 <= key <= glfw.KEY_9 and action == glfw.PRESS:
            self.frame = int(self.total_frames * (key - glfw.KEY_0) * 0.1)
            glfw.set_time(self.frame / self.fps)
            
        if not self.playing:
            if key == glfw.KEY_LEFT_BRACKET and action == glfw.PRESS:
                self.frame = max(self.frame - 1, 0)
            elif key == glfw.KEY_RIGHT_BRACKET and action == glfw.PRESS:
                self.frame = min(self.frame + 1, self.total_frames - 1)
            if key == glfw.KEY_LEFT and action == glfw.PRESS:
                self.frame = max(self.frame - 10, 0)
            elif key == glfw.KEY_RIGHT and action == glfw.PRESS:
                self.frame = min(self.frame + 10, self.total_frames - 1)
            glfw.set_time(self.frame / self.fps)
        
        # F6: record
        if key == glfw.KEY_F6 and action == glfw.PRESS:
            if self.recording:
                self.record_end_time = time.perf_counter()
                self.save_video()
                self.captures = []
            else:
                self.record_start_time = time.perf_counter()
            self.recording = not self.recording


    def update(self):
        # time setting
        if self.playing:
            self.frame = min(int(glfw.get_time() * self.fps), self.total_frames - 1)
        else:
            glfw.set_time(self.frame / self.fps)

        # stop playing at the end of the motion
        if self.playing and self.frame == self.total_frames - 1:
            self.playing = False

    def late_update(self):
        if self.recording:
            self.captures.append(super().capture_screen())

    """ Capture functions """
    def save_video(self):
        video_dir = os.path.join(self.capture_path, "videos")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        
        video_path = os.path.join(video_dir, datetime.datetime.now().strftime("%H-%M-%S") + ".mp4")
        fps = len(self.captures) / (self.record_end_time - self.record_start_time)
        height, width, _ = self.captures[0].shape
        
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for image in self.captures:
            video.write(image)
        video.release()

        glfw.set_time(self.frame / self.fps)

""" Class for motion data visualization """
class MotionApp(AnimApp):
    def __init__(self, motion: Motion, model: Model=None, skeleton_dict=LAFAN_BVH_TO_FBX):
        super().__init__(len(motion), motion.fps)
        self.motion = motion
        self.model = model
        if self.model is not None:
            if skeleton_dict is None:
                raise ValueError("skeleton_dict must be provided if model is provided")
            self.model.set_source_skeleton(self.motion.skeleton, skeleton_dict)

        self.frame = 0
        self.playing = True
        self.recording = False
        self.record_start_time = 0
        self.record_end_time = 0
        self.captures = []

        self.grid = Render.plane().set_scale(50).set_uv_repeat(5).set_texture("grid.png")
        self.axis = Render.axis()
        self.text = Render.text()
    
    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        # render options
        if key == glfw.KEY_G and action == glfw.PRESS:
            self.grid.switch_visible()
        if key == glfw.KEY_A and action == glfw.PRESS:
            self.axis.switch_visible()
        if key == glfw.KEY_T and action == glfw.PRESS:
            self.text.switch_visible()

    def late_update(self):
        super().late_update()

        # render the current frame
        self.text.set_text(self.frame).draw()
        
    def render(self, render_model=True, render_xray=False):
        super().render()

        # render the environment
        self.grid.draw()
        self.axis.draw()

        # render the model
        if render_model is True:
            if self.model is not None:
                self.model.set_pose_by_source(self.motion.poses[self.frame])
                Render.model(self.model).draw()

        # render the xray
        if render_xray:
            self.render_xray(self.motion.poses[self.frame])

    def render_xray(self, pose, albedo=[0, 0, 0]):
        if not hasattr(self, "joint_sphere") or not hasattr(self, "joint_bone"):
            self.joint_sphere = Render.sphere(0.03)
            self.joint_bone   = Render.cone(radius=0.03, height=1, sectors=4)
        
        global_p = pose.global_p
        # for i in range(pose.skeleton.num_joints):
        #     self.joint_sphere.set_position(global_p[i]).set_albedo([0, 1, 1]).draw()

        glDisable(GL_DEPTH_TEST)
        for i in range(1, self.motion.skeleton.num_joints):
            parent_pos = global_p[pose.skeleton.parent_idx[i]]

            center = glm.vec3((parent_pos + global_p[i]) / 2)
            dist = np.linalg.norm(parent_pos - global_p[i])
            dir = glm.vec3((global_p[i] - parent_pos) / (dist + 1e-8))

            axis = glm.cross(glm.vec3(0, 1, 0), dir)
            angle = glm.acos(glm.dot(glm.vec3(0, 1, 0), dir))
            orientation = glm.rotate(glm.mat4(1.0), angle, axis)
            
            self.joint_bone.set_position(center).set_orientation(orientation).set_scale(glm.vec3(1.0, dist, 1.0)).set_albedo(albedo).set_color_mode(True).draw()
        glEnable(GL_DEPTH_TEST)