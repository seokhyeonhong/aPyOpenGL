from pymovis.vis.app import App
from pymovis.vis.appmanager import AppManager
from pymovis.vis.render import Render
from OpenGL.GL import *
import glm

class MyApp(App):
    def __init__(self):
        super().__init__()
        self.primitives = []
        self.primitives.append(Render.arrow())
        self.primitives.append(Render.plane().set_texture("example.png").set_scale(50).set_uv_repeat(5))
        self.primitives.append(Render.sphere().set_position(glm.vec3(3,0.5,3)).set_texture("example.png"))
        self.primitives.append(Render.cube().set_position(1, 0.5, 1))
        self.primitives.append(Render.text("Rendering").set_position(3, 0, 0))
        self.primitives.append(Render.cubemap("skybox"))

    def render(self):
        for p in self.primitives:
            p.draw()

if __name__ == "__main__":
    app_manager = AppManager.initialize()
    app = MyApp()
    app_manager.run(app)