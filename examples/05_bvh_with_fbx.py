import os
import glfw
import glm

from aPyOpenGL import agl

BVH2FBX = {
    "Hips": "mixamorig:Hips",
    "Spine": "mixamorig:Spine",
    "Spine1": "mixamorig:Spine1",
    "Spine2": "mixamorig:Spine2",
    "Neck": "mixamorig:Neck",
    "Head": "mixamorig:Head",
    "LeftShoulder": "mixamorig:LeftShoulder",
    "LeftArm": "mixamorig:LeftArm",
    "LeftForeArm": "mixamorig:LeftForeArm",
    "LeftHand": "mixamorig:LeftHand",
    "LeftHandThumb1": "mixamorig:LeftHandThumb1",
    "LeftHandThumb2": "mixamorig:LeftHandThumb2",
    "LeftHandThumb3": "mixamorig:LeftHandThumb3",
    "LeftHandIndex1": "mixamorig:LeftHandIndex1",
    "LeftHandIndex2": "mixamorig:LeftHandIndex2",
    "LeftHandIndex3": "mixamorig:LeftHandIndex3",
    "LeftHandMiddle1": "mixamorig:LeftHandMiddle1",
    "LeftHandMiddle2": "mixamorig:LeftHandMiddle2",
    "LeftHandMiddle3": "mixamorig:LeftHandMiddle3",
    "LeftHandRing1": "mixamorig:LeftHandRing1",
    "LeftHandRing2": "mixamorig:LeftHandRing2",
    "LeftHandRing3": "mixamorig:LeftHandRing3",
    "LeftHandPinky1": "mixamorig:LeftHandPinky1",
    "LeftHandPinky2": "mixamorig:LeftHandPinky2",
    "LeftHandPinky3": "mixamorig:LeftHandPinky3",
    "RightShoulder": "mixamorig:RightShoulder",
    "RightArm": "mixamorig:RightArm",
    "RightForeArm": "mixamorig:RightForeArm",
    "RightHand": "mixamorig:RightHand",
    "RightHandThumb1": "mixamorig:RightHandThumb1",
    "RightHandThumb2": "mixamorig:RightHandThumb2",
    "RightHandThumb3": "mixamorig:RightHandThumb3",
    "RightHandIndex1": "mixamorig:RightHandIndex1",
    "RightHandIndex2": "mixamorig:RightHandIndex2",
    "RightHandIndex3": "mixamorig:RightHandIndex3",
    "RightHandMiddle1": "mixamorig:RightHandMiddle1",
    "RightHandMiddle2": "mixamorig:RightHandMiddle2",
    "RightHandMiddle3": "mixamorig:RightHandMiddle3",
    "RightHandRing1": "mixamorig:RightHandRing1",
    "RightHandRing2": "mixamorig:RightHandRing2",
    "RightHandRing3": "mixamorig:RightHandRing3",
    "RightHandPinky1": "mixamorig:RightHandPinky1",
    "RightHandPinky2": "mixamorig:RightHandPinky2",
    "RightHandPinky3": "mixamorig:RightHandPinky3",
    "LeftUpLeg": "mixamorig:LeftUpLeg",
    "LeftLeg": "mixamorig:LeftLeg",
    "LeftFoot": "mixamorig:LeftFoot",
    "LeftToeBase": "mixamorig:LeftToeBase",
    "RightUpLeg": "mixamorig:RightUpLeg",
    "RightLeg": "mixamorig:RightLeg",
    "RightFoot": "mixamorig:RightFoot",
    "RightToeBase": "mixamorig:RightToeBase",
}

class MotionApp(agl.App):
    def __init__(self, bvh_filename, fbx_filename):
        super().__init__()

        # motion data
        bvh = agl.BVH(bvh_filename)
        fbx = agl.FBX(fbx_filename)
        self.motion = bvh.motion()
        self.model = fbx.model()
        self.model.set_joint_map(BVH2FBX)

        self.total_frames = self.motion.num_frames
    
    def update(self):
        super().update()
        self.model.set_pose(self.motion.poses[self.frame % self.total_frames])
    
    def render(self):
        super().render()
        agl.Render.model(self.model).draw()

    def render_xray(self):
        super().render_xray()
        agl.Render.skeleton(self.motion.poses[self.frame % self.total_frames]).draw()
    
    def render_text(self):
        super().render_text()
        agl.Render.text_on_screen(f"Frame: {self.frame % self.total_frames} / {self.total_frames}").draw()

if __name__ == "__main__":
    bvh_filename = os.path.join(agl.AGL_PATH, "data/bvh/ybot_capoeira.bvh")
    fbx_filename = os.path.join(agl.AGL_PATH, "data/fbx/model/ybot.fbx")
    agl.AppManager.start(MotionApp(bvh_filename, fbx_filename))