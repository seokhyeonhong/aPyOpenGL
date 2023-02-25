import numpy as np
from pymovis.motion.data import bvh, fbx
from pymovis.vis import MotionApp, AppManager, Render

def env_sensor(radius, num):
    x, z = np.meshgrid(np.linspace(-radius, radius, num), np.linspace(-radius, radius, num))
    y = np.zeros_like(x)
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)

class MyApp(MotionApp):
    def __init__(self, motion, model):
        super().__init__(motion, model)
        self.motion = motion
        self.model = model
        self.sensor = env_sensor(1, 11) # (S, S, 3)
        self.sensor_sphere = Render.sphere(0.05).set_albedo([0.2, 1, 0.2])
        self.base_sphere = Render.sphere(0.05).set_albedo([1, 0.2, 0.2])
    def render(self):
        super().render(render_xray=False)

        # transform sensor
        forward = self.motion.poses[self.frame].forward
        up = self.motion.poses[self.frame].up
        left = self.motion.poses[self.frame].left
        R = np.stack([left, up, forward], axis=-1)
        sensor = np.einsum("ij,aj->ai", R, self.sensor) + self.motion.poses[self.frame].base
        self.base_sphere.draw()
        for s in sensor:
            self.sensor_sphere.set_position(s).draw()

if __name__ == "__main__":
    # app cycle manager
    app_manager = AppManager()

    # load data
    motion = bvh.load("data/motion.bvh", v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01)
    model = fbx.FBX("data/character.fbx").model()

    # align and slice
    motion.align_by_frame(600, origin_axes="xz")
    motion = motion.make_window(600, 1000)

    # create app
    app = MyApp(motion, model)

    # run app
    app_manager.run(app)