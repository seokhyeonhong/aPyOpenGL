from pymovis.vis.app import App
from pymovis.vis.appmanager import AppManager
from pymovis.vis.render import Render
from pymovis.motion.data.fbx import FBX
from OpenGL.GL import *
import glm

class MyApp(App):
    def __init__(self, model):
        super().__init__()
        self.model = Render.model(model)

    def render(self):
        self.model.draw()

if __name__ == "__main__":
    app_manager = AppManager()
    model = FBX("./data/test.fbx", scale=1).model()
    app = MyApp(model)
    app_manager.run(app)