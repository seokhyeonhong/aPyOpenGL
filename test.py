from pymovis.vis.app import App
from pymovis.vis.appmanager import AppManager
from pymovis.vis.render import Render
from pymovis.motion.data.fbx import FBX
from OpenGL.GL import *
import glm

class MyApp(App):
    def __init__(self):
        super().__init__()
        self.grid = Render.grid().set_grid_color(0.3, 0.7).set_scale(100)
        self.sphere = Render.sphere()

    def render(self):
        self.grid.draw()
        self.sphere.draw()

if __name__ == "__main__":
    app_manager = AppManager()
    app = MyApp()
    app_manager.run(app)