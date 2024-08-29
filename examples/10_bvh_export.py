import os
import glfw
import glm
import numpy as np

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
    def __init__(self, bvh_filename1, bvh_filename2, fbx_filename):
        super().__init__()

        # motion data
        bvh1 = agl.BVH(bvh_filename1)
        bvh2 = agl.BVH(bvh_filename2)
        fbx = agl.FBX(fbx_filename)

        self.motion1 = bvh1.motion()
        self.motion2 = bvh2.motion()
        self.model1 = fbx.model()
        self.model2 = fbx.model()
        self.model1.set_joint_map(BVH2FBX)
        self.model2.set_joint_map(BVH2FBX)

        self.total_frames = self.motion1.num_frames
    
    def update(self):
        super().update()
        
        # Move pose by 1.0 in x-axis
        pose1 = self.motion1.poses[self.frame % self.total_frames]
        self.render_pose1 = agl.motion.Pose(pose1.skeleton, pose1.local_quats, pose1.root_pos)
        self.model1.set_pose(self.render_pose1)

        pose2 = self.motion2.poses[self.frame % self.total_frames]
        self.render_pose2 = agl.motion.Pose(pose2.skeleton, pose2.local_quats, pose2.root_pos + np.array([1.0, 0, 0]))
        self.model2.set_pose(self.render_pose2)
    
    def render(self):
        super().render()
        agl.Render.model(self.model1).draw()
        agl.Render.model(self.model2).draw() # * model to draw

    def render_xray(self):
        super().render_xray()
        agl.Render.skeleton(self.render_pose1).draw()
        agl.Render.skeleton(self.render_pose2).draw() # * pose to draw

        # TODO: Render two characters

    
    def render_text(self):
        super().render_text()
        agl.Render.text_on_screen(f"Frame: {self.frame % self.total_frames} / {self.total_frames}").draw()

if __name__ == "__main__":
    bvh_filename1 = os.path.join(agl.AGL_PATH, "data/bvh/ybot_capoeira.bvh")
    bvh_filename2 = os.path.join(agl.AGL_PATH, "data/bvh/ybot_capoeira_export.bvh")
    fbx_filename = os.path.join(agl.AGL_PATH, "data/fbx/model/ybot.fbx")

    # Import bvh
    bvh = agl.BVH(bvh_filename1)
    fbx = agl.FBX(fbx_filename)
    motion = bvh.motion()

    # Export bvh
    motion.export_as_bvh(bvh_filename2)

    agl.AppManager.start(MotionApp(bvh_filename1, bvh_filename2, fbx_filename))