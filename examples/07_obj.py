import os
import glm

from pymovis.vis import App, AppManager, Render

class MyApp(App):
    def __init__(self, filename):
        super().__init__()
        self.obj = Render.obj(filename)

    def render(self):
        super().render()
        self.obj.draw()

if __name__ == "__main__":
    filename = os.path.join(os.path.dirname(__file__), "../data/obj/teapot.obj")
    AppManager.start(MyApp(filename))