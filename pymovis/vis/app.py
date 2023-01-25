from OpenGL.GL import *

import glfw
import os
import datetime
import cv2
import numpy as np
import glm

from pymovis.motion.core import Motion
from pymovis.motion.ops import npmotion

from pymovis.vis.camera import Camera
from pymovis.vis.light import DirectionalLight, PointLight
from pymovis.vis.render import Render
from pymovis.vis.model import Model
from pymovis.vis.const import LAFAN_BVH_TO_FBX

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

""" Class for motion data visualization """
class MotionApp(App):
    def __init__(self, motion: Motion, model: Model=None, skeleton_dict=LAFAN_BVH_TO_FBX):
        super().__init__()
        self.motion = motion
        self.model = model
        if self.model is not None:
            if skeleton_dict is None:
                raise ValueError("skeleton_dict must be provided if model is provided")
            self.model.set_source_skeleton(self.motion.skeleton, skeleton_dict)

        self.frame = 0
        self.prev_frame = -1
        self.playing = True
        self.recording = False
        self.captures = []

        self.grid = Render.plane().set_scale(50).set_uv_repeat(5).set_texture("grid.png")
        self.axis = Render.axis()
        self.text = Render.text()
    
    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        # play / pause
        if key == glfw.KEY_SPACE and action == glfw.PRESS:
            self.playing = not self.playing
            if self.playing and self.frame == len(self.motion) - 1:
                self.frame = 0
                glfw.set_time(0)
        
        # move frames
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
        
        # render and capture options
        if action == glfw.PRESS:
            if key == glfw.KEY_G:
                self.grid.switch_visible()
            elif key == glfw.KEY_A:
                self.axis.switch_visible()
            elif key == glfw.KEY_T:
                self.text.switch_visible()
            elif key == glfw.KEY_F6:
                if self.recording:
                    self.save_video(self.captures)
                    self.captures = []
                self.recording = not self.recording

    def update(self):
        # stop playing at the end of the motion
        if self.playing and self.frame == len(self.motion) - 1:
            self.playing = False

    def late_update(self):
        # rendering the current frame
        self.text.set_text(self.frame).draw()

        # recording
        if self.recording:
            if self.prev_frame == self.frame and self.playing:
                return
                
            self.captures.append(super().capture_screen())
            self.prev_frame = self.frame
        
    def render(self, render_xray=False):
        # time setting
        if self.playing:
            self.frame = min(int(glfw.get_time() * self.motion.fps), len(self.motion) - 1)
        else:
            glfw.set_time(self.frame / self.motion.fps)
        
        # render the current frame
        self.grid.draw()
        self.axis.draw()
        self.motion.render_by_frame(self.frame)

        if self.model is not None:
            self.model.set_pose_by_source(self.motion.poses[self.frame])
            Render.model(self.model).draw()
        else:
            self.render_xray(self.motion.poses[self.frame])

        if render_xray:
            self.render_xray(self.motion.poses[self.frame])

    def render_xray(self, pose):
        glDisable(GL_DEPTH_TEST)
        if not hasattr(self, "joint_sphere") or not hasattr(self, "joint_bone"):
            self.joint_sphere = Render.sphere(0.03)
            self.joint_bone   = Render.cylinder(0.03, 1.0)
        
        global_R, global_p = npmotion.R_fk(pose.local_R, pose.root_p, pose.skeleton)
        for i in range(pose.skeleton.num_joints):
            self.joint_sphere.set_position(global_p[i]).set_albedo([0, 1, 1]).draw()

        for i in range(1, self.motion.skeleton.num_joints):
            parent_pos = global_p[pose.skeleton.parent_idx[i]]

            center = glm.vec3((parent_pos + global_p[i]) / 2)
            dist = np.linalg.norm(parent_pos - global_p[i])
            dir = glm.vec3((global_p[i] - parent_pos) / (dist + 1e-8))

            axis = glm.cross(glm.vec3(0, 1, 0), dir)
            angle = glm.acos(glm.dot(glm.vec3(0, 1, 0), dir))
            rotation = glm.rotate(glm.mat4(1.0), angle, axis)
            
            self.joint_bone.set_position(center).set_orientation(rotation).set_scale(glm.vec3(1.0, dist, 1.0)).set_albedo([0, 1, 1]).draw()
        glEnable(GL_DEPTH_TEST)

    def terminate(self):
        self.motion.__delattr__("joint_sphere")
        self.motion.__delattr__("joint_bone")

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