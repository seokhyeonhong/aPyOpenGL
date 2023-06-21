from pymovis.motion.data.fbx import FBX
from pymovis.vis import App, AppManager, Render

import glm, glfw

class MyApp(App):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def render(self):
        Render.model(self.model).draw()

if __name__ == "__main__":
    # app cycle manager
    app_manager = AppManager()

    # load data
    fbx = FBX("data/character.fbx")

    # create app
    app = MyApp(fbx.model())

    # run app
    app_manager.run(app)