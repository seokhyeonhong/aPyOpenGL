import os
import glm

from pymovis import vis

class MyApp(vis.App):
    def __init__(self, filename):
        super().__init__()
        self.obj = vis.Render.obj(filename)

    def render(self):
        super().render()
        self.obj.draw()

if __name__ == "__main__":
    filename = os.path.join(os.path.dirname(__file__), "../data/obj/teapot.obj")
    vis.AppManager.start(MyApp(filename))