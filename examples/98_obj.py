import os
import glm

from pymovis import agl

class MyApp(agl.App):
    def __init__(self, filename):
        super().__init__()
        self.obj = agl.Render.obj(filename)

    def render(self):
        super().render()
        self.obj.draw()

if __name__ == "__main__":
    filename = os.path.join(agl.AGL_PATH, "data/obj/teapot.obj")
    agl.AppManager.start(MyApp(filename))