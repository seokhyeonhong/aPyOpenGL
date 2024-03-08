# import os
# import glm
# import glfw
# import numpy as np

# from aPyOpenGL import agl, kin

# class MotionApp(agl.App):
#     def __init__(self, motion_filename, model_filename):
#         super().__init__()

#         # motion data
#         self.motion = agl.FBX(motion_filename).motions()[0]
#         self.model  = agl.FBX(model_filename).model()
#         self.total_frames = self.motion.num_frames
#         self.fps = self.motion.fps
    
#     def start(self):
#         super().start()
#         self.render_model = agl.Render.model(self.model)
#         self.kinpose = kin.KinPose(self.motion.poses[0])

#     def update(self):
#         super().update()
        
#         curr_frame = self.frame % self.total_frames
#         self.kinpose.set_pose(self.motion.poses[curr_frame])

#         # update model to render
#         self.model.set_pose(self.kinpose.to_pose())

#     def render(self):
#         super().render()
#         self.render_model.update_model(self.model).draw()

# if __name__ == "__main__":
#     motion_filename = os.path.join(agl.AGL_PATH, "data/fbx/motion/ybot_walking.fbx")
#     model_filename  = os.path.join(agl.AGL_PATH, "data/fbx/model/ybot.fbx")
#     agl.AppManager.start(MotionApp(motion_filename, model_filename))