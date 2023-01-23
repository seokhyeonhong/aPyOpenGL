import torch
import glfw
import glm

from pymovis.motion.data import bvh
from pymovis.motion.data.fbx import FBX
from pymovis.motion.core import Motion
from pymovis.vis.render import Render
from pymovis.vis.app import MotionApp
from pymovis.vis.model import Model
from pymovis.vis.appmanager import AppManager

BVH_TO_FBX = {
    "Hips":          "Model:Hips",
    "LeftUpLeg":     "Model:LeftUpLeg",
    "LeftLeg":       "Model:LeftLeg",
    "LeftFoot":      "Model:LeftFoot",
    "LeftToe":       "Model:LeftToe",
    "RightUpLeg":    "Model:RightUpLeg",
    "RightLeg":      "Model:RightLeg",
    "RightFoot":     "Model:RightFoot",
    "RightToe":      "Model:RightToe",
    "Spine":         "Model:Spine",
    "Spine1":        "Model:Spine1",
    "Spine2":        "Model:Spine2",
    "Neck":          "Model:Neck",
    "Head":          "Model:Head",
    "LeftShoulder":  "Model:LeftShoulder",
    "LeftArm":       "Model:LeftArm",
    "LeftForeArm":   "Model:LeftForeArm",
    "LeftHand":      "Model:LeftHand",
    "RightShoulder": "Model:RightShoulder",
    "RightArm":      "Model:RightArm",
    "RightForeArm":  "Model:RightForeArm",
    "RightHand":     "Model:RightHand",
}

class MyApp(MotionApp):
    def __init__(self, model: Model, motion: Motion):
        super().__init__(motion)
        model.set_source_skeleton(motion.skeleton, BVH_TO_FBX)
        self.model = model
        self.motion = motion

    def render(self):
        super().render()
        pose = self.motion.poses[self.frame]
        self.model.set_pose_by_source(pose)
        Render.model(self.model).draw()

if __name__ == "__main__":
    app_manager = AppManager()
    # fbx = FBX("./data/ybot.fbx")
    fbx = FBX("D:/data/LaFAN1/model_skeleton.fbx")
    motion = bvh.load("D:/data/LaFAN1/walk3_subject1.bvh", v_forward=[0, 1, 0], v_up=[0, 0, 1])
    motion.align_by_frame(0)
    model = fbx.model()
    app = MyApp(model, motion)
    app_manager.run(app)