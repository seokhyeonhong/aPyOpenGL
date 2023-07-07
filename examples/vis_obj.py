import os
import numpy as np

from pymovis.vis import App, AppManager, Render

class ObjApp(App):
    def __init__(self):
        super().__init__()
        filepath = os.path.join(os.path.dirname(__file__), "../data/obj/teapot.obj")
        self.character = Render.obj(filepath, scale=0.01).set_background(0.1)

    def render(self):
        super().render()
        self.character.draw()

if __name__ == "__main__":
    app_manager = AppManager()
    app = ObjApp()
    app_manager.run(app)