import os
import numpy as np
import glfw
from OpenGL.GL import *

from pymovis.motion import BVH, FBX
from pymovis.vis import MotionApp, AppManager, Render, YBOT_FBX_DICT

def env_sensor(radius, num):
    x, z = np.meshgrid(np.linspace(-radius, radius, num), np.linspace(-radius, radius, num))
    y = np.zeros_like(x)
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)

class MyApp(MotionApp):
    def __init__(self, motion, model, joint_ids, contacts):
        super().__init__(motion, model, YBOT_FBX_DICT)
        self.motion = motion
        self.model = model

        # sensor
        self.show_sensor = True
        self.sensor = env_sensor(1, 11) # (S, S, 3)
        self.sensor_sphere = Render.sphere(0.05).set_albedo([0.2, 1, 0.2]).set_background(0.1)

        # contacts
        self.show_contacts = True
        self.joint_ids = joint_ids
        self.contacts = contacts
        self.contact_sphere = Render.sphere(0.05).set_albedo([1, 0, 0])

    def render(self):
        super().render()

        # transform sensor
        if self.show_sensor:
            forward = self.motion.poses[self.frame].forward
            up = self.motion.poses[self.frame].up
            left = self.motion.poses[self.frame].left
            R = np.stack([left, up, forward], axis=-1)
            sensor = np.einsum("ij,aj->ai", R, self.sensor) + self.motion.poses[self.frame].base
            for s in sensor:
                self.sensor_sphere.set_position(s).draw()
        
        # contacts
        if self.show_contacts:
            contact = self.contacts[self.frame]
            for i, j in enumerate(self.joint_ids):
                albedo = [0, 1, 0] if contact[i] else [1, 0, 0]
                glDisable(GL_DEPTH_TEST)
                self.contact_sphere.set_position(self.motion.poses[self.frame].global_p[j]).set_albedo(albedo).draw()
                glEnable(GL_DEPTH_TEST)
    
    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        if action == glfw.PRESS:
            if   key == glfw.KEY_Q: self.show_sensor = not self.show_sensor
            elif key == glfw.KEY_W: self.show_contacts = not self.show_contacts

if __name__ == "__main__":
    # app cycle manager
    app_manager = AppManager()

    # load data
    motion_path = os.path.join(os.path.dirname(__file__), "../data/bvh/ybot_walk.bvh")
    model_path = os.path.join(os.path.dirname(__file__), "../data/fbx/model/ybot.fbx")
    motion = BVH.load(motion_path, v_forward=[0, 0, 1], v_up=[0, 1, 0], to_meter=0.01)
    model = FBX(model_path).model()

    # align and slice
    motion.align_by_frame(250, origin_axes="xz")

    # contacts
    joint_names = ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]
    joint_ids = [motion.skeleton.idx_by_name[name] for name in joint_names]
    contacts = []
    for i in range(1, len(motion.poses)):
        curr_p = motion.poses[i].global_p[joint_ids]
        prev_p = motion.poses[i-1].global_p[joint_ids]
        feet_v = np.sum((curr_p - prev_p)**2, axis=-1) # squared norm
        contact = (feet_v < 2e-4)
        contacts.append(contact)

    contacts = np.array(contacts)
    contacts = np.concatenate([contacts[0:1], contacts], axis=0)

    # create app
    app = MyApp(motion, model, joint_ids, contacts)

    # run app
    app_manager.run(app)