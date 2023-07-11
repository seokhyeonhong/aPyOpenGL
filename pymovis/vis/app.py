from enum import Enum
from OpenGL.GL import *

import glfw
import os
import datetime
import cv2
import numpy as np
import glm

from .motion import Motion
from .camera import Camera
from .light  import Light, DirectionalLight, PointLight
from .render import Render
from .model  import Model
from .ui     import UI

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
        self.window = self.init_glfw()

    class IO:
        def __init__(self):
            self.last_mouse_x = 0
            self.last_mouse_y = 0
            self.mouse_middle_down = False
            self.mouse_left_down = False
    
    def init_glfw(self):
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)

        # create window
        window = glfw.create_window(self.width, self.height, "pymovis", None, None)
        glfw.make_context_current(window)

        if not window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")
        
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        # callbacks
        glfw.set_framebuffer_size_callback(window, self.on_resize)
        glfw.set_key_callback(window, self.key_callback)
        glfw.set_cursor_pos_callback(window, self.mouse_callback)
        glfw.set_mouse_button_callback(window, self.mouse_button_callback)
        glfw.set_scroll_callback(window, self.scroll_callback)
        glfw.set_error_callback(self.on_error)
        
        # global OpenGL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_LINE_SMOOTH)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthFunc(GL_LEQUAL)
        glCullFace(GL_BACK)
        # glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # intialize shaders
        Render.initialize_shaders()
        glViewport(0, 0, self.width, self.height)

        return window

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

    def render_xray(self):
        pass
    
    def terminate(self):
        pass

    """ Callback functions for glfw and camera control """
    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            Render.clear()
        elif key == glfw.KEY_F1 and action == glfw.PRESS:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        elif key == glfw.KEY_F2 and action == glfw.PRESS:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

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

    def initialize_ui(self):
        self.ui.initialize(self.window)
    
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

    def __init__(self):
        super().__init__()
        self.total_frames = 0
        self.fps          = 30
        self.curr_frame   = 0
        self.prev_frame   = 0
        self.playing      = True
        self.record_mode  = AnimApp.RecordMode.eNONE
        self.captures     = []

    def start(self):
        super().start()

        # render options
        self.axis = Render.axis()
        self.grid = Render.plane(200, 200).albedo(0.15).floor(True)
        self.text = Render.text_on_screen().position([50, 50, 0])

        # UI options
        self.ui.add_menu("AnimApp")
        self.ui.add_menu_item("AnimApp", "Axis", self.axis.switch_visible, key=glfw.KEY_A)
        self.ui.add_menu_item("AnimApp", "Grid", self.grid.switch_visible, key=glfw.KEY_G)
        self.ui.add_menu_item("AnimApp", "Text", self.text.switch_visible, key=glfw.KEY_T)
        self.ui.add_menu_item("AnimApp", "Play/Pause", self.toggle_play, key=glfw.KEY_SPACE)

        # glfw
        glfw.set_time(0)
    
    def toggle_play(self):
        self.playing = not self.playing

        # Replay when the end of the animation is reached
        if self.playing and self.curr_frame == self.total_frames - 1:
            self.curr_frame = 0
            glfw.set_time(0)
    
    def move_frame(self, offset):
        self.curr_frame = max(0, min(self.curr_frame + offset, self.total_frames - 1))
        glfw.set_time(self.curr_frame / self.fps)

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        if action != glfw.PRESS:
            return
        
        # 0~9, []: frame control
        if glfw.KEY_0 <= key <= glfw.KEY_9:
            self.curr_frame = int(self.total_frames * (key - glfw.KEY_0) * 0.1)
        
        # F6/F7/F8: record section / all frames / section to images
        elif key == glfw.KEY_F6:
            if self.record_mode == AnimApp.RecordMode.eSECTION_TO_VID:
                self.save_video()
                self.captures = []
                self.record_mode = AnimApp.RecordMode.eNONE
            elif self.record_mode == AnimApp.RecordMode.eNONE:
                self.record_mode = AnimApp.RecordMode.eSECTION_TO_VID
        
        # frame control
        elif key == glfw.KEY_LEFT_BRACKET:
            self.move_frame(-1)
        elif key == glfw.KEY_RIGHT_BRACKET:
            self.move_frame(+1)
        elif key == glfw.KEY_LEFT:
            self.move_frame(-10)
        elif key == glfw.KEY_RIGHT:
            self.move_frame(+10)

    def update(self):
        super().update()

        # time setting
        if self.record_mode == AnimApp.RecordMode.eNONE:
            if self.playing:
                self.curr_frame = min(int(glfw.get_time() * self.fps), self.total_frames - 1)
            else:
                glfw.set_time(self.curr_frame / self.fps)
        else:
            self.curr_frame = min(int(glfw.get_time() * self.fps), self.total_frames - 1)
            if self.curr_frame - self.prev_frame > 1:
                self.curr_frame = self.prev_frame + 1
                glfw.set_time(self.curr_frame / self.fps)
                
        # stop playing at the end of the motion
        if self.playing and self.curr_frame == self.total_frames - 1:
            self.playing = False
    
    def render(self):
        super().render()
        self.grid.draw()
        self.axis.draw()
    
    def render_text(self):
        super().render_text()
        self.text.text(f"Frame: {self.curr_frame + 1} / {self.total_frames}").draw()

    def late_update(self):
        super().late_update()

        # capture video
        if self.record_mode != AnimApp.RecordMode.eNONE:
            if (self.curr_frame - self.prev_frame) == 1:
                self.captures.append(super().capture_screen())
            if self.record_mode == AnimApp.RecordMode.eSECTION_TO_VID and self.curr_frame == self.total_frames - 1:
                self.save_video()
                self.captures = []
                self.record_mode = AnimApp.RecordMode.eNONE

        # update previous frame
        self.prev_frame = self.curr_frame

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

        glfw.set_time(self.curr_frame / self.fps)
    
    def save_images(self):
        image_dir = os.path.join(self.capture_path, f"images-{datetime.datetime.now().strftime('%H-%M-%S')}")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        
        for i, image in enumerate(self.captures):
            image_path = os.path.join(image_dir, "{:04d}.png".format(i))
            cv2.imwrite(image_path, image)

""" Class for motion data visualization """
class MotionApp(AnimApp):
    def __init__(self):
        super().__init__()

        # motion data
        self.motion: Motion = None
        self.model: Model = None

        # camera options
        self.focus_on_root = False
        self.follow_root = False
        self.init_cam_pos = self.camera.position

    def start(self):
        super().start()

        if self.motion is None:
            raise ValueError("Motion data is not loaded.")
        
        if self.model is None:
            self.model = Model(meshes=None, skeleton=self.motion.get_pose_at(0).get_skeleton())

        self.total_frames = self.motion.num_frames()
        self.fps = self.motion.get_fps()

        # render options
        self.render_model    = Render.model(self.model)
        self.render_skeleton = Render.skeleton(self.model)
        
        # UI options
        self.ui.add_menu("MotionApp")
        self.ui.add_menu_item("MotionApp", "X-Ray", self.render_skeleton.switch_visible, key=glfw.KEY_X)
        if self.render_model is not None:
            self.ui.add_menu_item("MotionApp", "Model", self.render_model.switch_visible, key=glfw.KEY_M)
    
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
            self.camera.set_focus_position(self.motion.get_pose_at(self.curr_frame).get_root_p())
            self.camera.set_up(glm.vec3(0, 1, 0))
        elif self.follow_root:
            self.camera.set_position(self.motion.get_pose_at(self.curr_frame).get_root_p() + glm.vec3(0, 1.5, 5))
            self.camera.set_focus_position(self.motion.get_pose_at(self.curr_frame).get_root_p())
            self.camera.set_up(glm.vec3(0, 1, 0))
        self.camera.update()
        
        # set pose
        self.model.set_pose(self.motion.get_pose_at(self.curr_frame))

    def render(self):
        super().render()

        # render the environment
        self.grid.draw()
        self.axis.draw()

        # render the model
        if self.model is not None:
            self.render_model.buffer_xforms(self.model).draw()

    def render_xray(self):
        super().render_xray()
        self.render_skeleton.pose(self.model.pose).draw()