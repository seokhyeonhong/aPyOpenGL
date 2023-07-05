from enum import Enum
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
from pymovis.vis.light  import Light, DirectionalLight, PointLight
from pymovis.vis.render import Render
from pymovis.vis.model  import Model
from pymovis.vis.const  import LAFAN1_FBX_DICT
from pymovis.vis.ui     import UI

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
        self.ui = UI()

        self.capture_path = os.path.join("capture", str(datetime.date.today()))

    class IO:
        def __init__(self):
            self.last_mouse_x = 0
            self.last_mouse_y = 0
            self.mouse_middle_down = False
            self.mouse_left_down = False
    
    def init_window(self, window):
        self.window = window
        self.width, self.height = glfw.get_window_size(self.window)
        self.ui.initialize(self.window)

    """ Override these methods to add custom rendering code """
    def start(self):
        pass

    def update(self):
        pass

    def late_update(self):
        pass

    def render(self):
        pass

    def render_text(self):
        pass
    
    def terminate(self):
        pass

    """ Callback functions for glfw and camera control """
    def key_callback(self, window, key, scancode, action, mods):
        self.ui.key_callback(window, key, scancode, action, mods)
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

    def scroll_callback(self, window, x_offset, y_offset):
        left_alt_pressed = (glfw.get_key(window, glfw.KEY_LEFT_ALT) == glfw.PRESS)
        if left_alt_pressed:
            self.camera.dolly(y_offset)
        else:
            self.camera.zoom(y_offset)

    def on_error(self, error, desc):
        pass

    def on_resize(self, window, width, height):
        glViewport(0, 0, width, height)
        self.width, self.height = width, height
    
    """ Capture functions """
    def capture_screen(self):
        viewport = glGetIntegerv(GL_VIEWPORT)
        x, y, *_ = viewport
        
        glReadBuffer(GL_FRONT)
        data = glReadPixels(x, y, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        pixels = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        pixels = np.flip(pixels[:-self.ui.get_menu_height()], axis=0)
        image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        return image
    
    def save_image(self, image):
        image_dir = os.path.join(self.capture_path, "images")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        
        image_path = os.path.join(image_dir, datetime.datetime.now().strftime("%H-%M-%S") + ".png")
        cv2.imwrite(image_path, image)
    
    """ UI functions """
    def process_inputs(self):
        self.ui.process_inputs()
    
    def render_ui(self):
        self.ui.render()
    
    def terminate_ui(self):
        self.ui.terminate()

""" Class for general animation with a fixed number of frames """
# TODO: refactor this class, focusing on glfw.set_time() and glfw.get_time()
class AnimApp(App):
    class RecordMode(Enum):
        eNONE           = 0
        eSECTION_TO_VID = 1
        # eSECTION_TO_IMG = 3

    def __init__(self, total_frames, fps=30):
        super().__init__()
        self.total_frames = int(total_frames)
        self.fps = fps

        self.frame = 0
        self.playing = True
        self.record_mode = AnimApp.RecordMode.eNONE
        self.captures = []

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        if action != glfw.PRESS:
            return
        
        # Space: play/pause
        if key == glfw.KEY_SPACE:
            self.playing = not self.playing
        
            # Replay when the end of the animation is reached
            if self.playing and self.frame == self.total_frames - 1:
                self.frame = 0

        # 0~9, []: frame control
        elif glfw.KEY_0 <= key <= glfw.KEY_9:
            self.frame = int(self.total_frames * (key - glfw.KEY_0) * 0.1)
        
        # F6/F7/F8: record section / all frames / section to images
        elif key == glfw.KEY_F6:
            if self.record_mode == AnimApp.RecordMode.eSECTION_TO_VID:
                self.save_video()
                self.captures = []
                self.record_mode = AnimApp.RecordMode.eNONE
            elif self.record_mode == AnimApp.RecordMode.eNONE:
                self.record_mode = AnimApp.RecordMode.eSECTION_TO_VID
        
        # frame control
        if not self.playing:
            if key == glfw.KEY_LEFT_BRACKET:
                self.frame = max(self.frame - 1, 0)
            elif key == glfw.KEY_RIGHT_BRACKET:
                self.frame = min(self.frame + 1, self.total_frames - 1)
            if key == glfw.KEY_LEFT:
                self.frame = max(self.frame - 10, 0)
            elif key == glfw.KEY_RIGHT:
                self.frame = min(self.frame + 10, self.total_frames - 1)
        
        # set GLFW time
        glfw.set_time(self.frame / self.fps)

    def update(self):
        # time setting
        if self.record_mode != AnimApp.RecordMode.eNONE:
            self.frame = min(self.frame + 1, self.total_frames - 1)
            glfw.set_time(self.frame / self.fps)
        elif self.playing:
            self.frame = min(int(glfw.get_time() * self.fps), self.total_frames - 1)
        else:
            glfw.set_time(self.frame / self.fps)

        # stop playing at the end of the motion
        if self.playing and self.frame == self.total_frames - 1:
            self.playing = False

    def late_update(self):
        if self.record_mode != AnimApp.RecordMode.eNONE:
            self.captures.append(super().capture_screen())

    """ Capture functions """
    def save_video(self):
        video_dir = os.path.join(self.capture_path, "videos")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        
        video_path = os.path.join(video_dir, datetime.datetime.now().strftime("%H-%M-%S") + ".mp4")
        height, width, _ = self.captures[0].shape
        
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        for image in self.captures:
            video.write(image)
        video.release()

        glfw.set_time(self.frame / self.fps)
    
    def save_images(self):
        image_dir = os.path.join(self.capture_path, f"images-{datetime.datetime.now().strftime('%H-%M-%S')}")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        
        for i, image in enumerate(self.captures):
            image_path = os.path.join(image_dir, "{:04d}.png".format(i))
            cv2.imwrite(image_path, image)

""" Class for motion data visualization """
class MotionApp(AnimApp):
    def __init__(self, motion: Motion, model: Model=None, skeleton_dict=LAFAN1_FBX_DICT):
        super().__init__(len(motion), motion.fps)
        self.motion = motion
        self.model = model
        if self.model is not None:
            if skeleton_dict is None:
                raise ValueError("skeleton_dict must be provided if model is provided")
            self.model.set_source_skeleton(self.motion.skeleton, skeleton_dict)

        # play options
        self.playing = True
        self.recording = False
        self.record_start_time = 0
        self.record_end_time = 0
        self.captures = []

        # camera options
        self.focus_on_root = False
        self.follow_root = False
        self.init_cam_pos = self.camera.position

        self.grid = Render.plane(200, 200).set_albedo(0.15).set_floor(True)
        self.axis = Render.axis().set_background(0.0)
        self.text = Render.text()

        # UI options
        self.ui.add_menu("MotionApp")
        self.ui.add_render_toggle("MotionApp", "Grid", self.grid, key=glfw.KEY_G, activated=True)
        self.ui.add_render_toggle("MotionApp", "Axis", self.axis, key=glfw.KEY_A, activated=True)
        self.ui.add_render_toggle("MotionApp", "Frame Text", self.text, key=glfw.KEY_T, activated=True)
    
    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        # set camera focus on the root
        if key == glfw.KEY_F3 and action == glfw.PRESS:
            self.focus_on_root = not self.focus_on_root
        elif key == glfw.KEY_F4 and action == glfw.PRESS:
            self.follow_root = not self.follow_root

    def update(self):
        super().update()

        # set camera focus on the root
        if self.focus_on_root:
            self.camera.set_focus_position(self.motion.poses[self.frame].root_p)
            self.camera.set_up(glm.vec3(0, 1, 0))
        elif self.follow_root:
            self.camera.set_position(self.motion.poses[self.frame].root_p + glm.vec3(0, 1.5, 5))
            self.camera.set_focus_position(self.motion.poses[self.frame].root_p)
            self.camera.set_up(glm.vec3(0, 1, 0))
        self.camera.update()

    def late_update(self):
        super().late_update()

        # render the current frame
        self.text.set_text(self.frame).draw()
        
    def render(self, render_model=True, render_xray=False, xray_color=[1, 0, 0], model_background=0.2):
        super().render()

        # render the environment
        self.grid.draw()
        self.axis.draw()

        # render the model
        if render_model is True:
            if self.model is not None:
                self.model.set_pose_by_source(self.motion.poses[self.frame])
                Render.model(self.model).set_background(model_background).draw()

        # render the xray
        if render_xray:
            self.render_xray(self.motion.poses[self.frame], xray_color)

    def render_xray(self, pose, albedo=[1, 0, 0], alpha=1.0):
        if not hasattr(self, "joint_sphere") or not hasattr(self, "joint_bone"):
            self.joint_sphere = Render.sphere(0.03)
            self.joint_bone   = Render.pyramid(radius=0.03, height=1, sectors=4)
        
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
            
            self.joint_bone.set_position(center).set_orientation(orientation).set_scale(glm.vec3(1.0, dist, 1.0)).set_albedo(albedo).set_alpha(alpha).draw()
        glEnable(GL_DEPTH_TEST)