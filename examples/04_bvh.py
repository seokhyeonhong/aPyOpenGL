import os
import glfw
import glm

from aPyOpenGL import agl

class MotionApp(agl.App):
    def __init__(self, bvh_filename):
        super().__init__()

        # motion data
        bvh = agl.BVH(bvh_filename)
        self.motion = bvh.motion()
        self.model = bvh.model()
        self.total_frames = self.motion.num_frames
        self.fps = self.motion.fps

        # camera options
        self.focus_on_root = False
        self.follow_root = False
        self.init_cam_pos = self.camera.position
    
    def update(self):
        super().update()

        # set camera focus on the root
        curr_frame = self.frame % self.total_frames
        if self.focus_on_root:
            self.camera.set_focus_position(self.motion.poses[curr_frame].root_p)
            self.camera.set_up(glm.vec3(0, 1, 0))
        elif self.follow_root:
            self.camera.set_position(self.motion.poses[curr_frame].root_p + glm.vec3(0, 1.5, 5))
            self.camera.set_focus_position(self.motion.poses[curr_frame].root_p)
            self.camera.set_up(glm.vec3(0, 1, 0))
        self.camera.update()
        
        # set pose
        self.model.set_pose(self.motion.poses[curr_frame])

    def render_xray(self):
        super().render_xray()
        agl.Render.skeleton(self.motion.poses[self.frame % self.total_frames]).draw()
    
    def render_text(self):
        super().render_text()
        agl.Render.text(f"Frame: {self.frame % self.total_frames} / {self.total_frames}").draw()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        # set camera focus on the root
        if key == glfw.KEY_F3 and action == glfw.PRESS:
            self.focus_on_root = not self.focus_on_root
        elif key == glfw.KEY_F4 and action == glfw.PRESS:
            self.follow_root = not self.follow_root
            
if __name__ == "__main__":
    bvh_filename = os.path.join(agl.AGL_PATH, "data/bvh/ybot_capoeira.bvh")
    agl.AppManager.start(MotionApp(bvh_filename))