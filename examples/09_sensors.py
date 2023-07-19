# import os
# import glm
# import glfw
# import numpy as np

# from aPyOpenGL import agl, kin

# class MotionApp(agl.AnimApp):
#     def __init__(self, motion_filename, model_filename):
#         super().__init__()

#         # motion data
#         self.motion = agl.FBX(motion_filename).motions()[0]
#         self.model  = agl.FBX(model_filename).model()
#         self.total_frames = self.motion.num_frames
#         self.fps = self.motion.fps
    
#     def start(self):
#         super().start()

#         # render options
#         self.render_skeleton = agl.Render.skeleton(self.model)
#         self.render_model = agl.Render.model(self.model)
#         self.render_sphere = agl.Render.sphere(0.05).albedo([1, 0, 0])

#         # kin pose
#         self.kinpose = kin.KinPose(self.motion.poses[0])

#         # sensor points
#         sensor_range = np.linspace(-1, 1, 10)
#         x, z = np.meshgrid(sensor_range, sensor_range)
#         self.sensor_points = np.stack([x.flatten(), np.zeros_like(x.flatten()), z.flatten(), np.ones_like(x.flatten())], axis=-1)

#     def update(self):
#         super().update()
        
#         # update kinpose basis to the origin
#         self.kinpose.set_pose(self.motion.poses[self.curr_frame])

#         # update model to render
#         self.model.set_pose(self.kinpose.to_pose())

#     def render(self):
#         super().render()
#         self.render_model.update_model(self.model).draw()

#         # draw sensor points
#         points = (self.kinpose.basis_xform @ self.sensor_points[..., None])[..., 0]
#         for point in points:
#             self.render_sphere.position(point[:3]).draw()

#     def render_xray(self):
#         super().render_xray()
#         self.render_skeleton.update_skeleton(self.model).draw()

# if __name__ == "__main__":
#     motion_filename = os.path.join(agl.AGL_PATH, "data/fbx/motion/ybot_walking.fbx")
#     model_filename  = os.path.join(agl.AGL_PATH, "data/fbx/model/ybot.fbx")
#     agl.AppManager.start(MotionApp(motion_filename, model_filename))