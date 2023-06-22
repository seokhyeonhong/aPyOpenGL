import numpy as np
from pymovis.motion import BVH, FBX
from pymovis.vis import App, AppManager, Render

class ObjApp(App):
    def __init__(self):
        super().__init__()
        self.character = Render.obj("data/character.obj", scale=0.01).set_background(0.1)

    def render(self):
        super().render()
        self.character.draw()

if __name__ == "__main__":
    app_manager = AppManager()
    app = ObjApp()
    app_manager.run(app)